import hashlib
import copy
import random
import numpy as np
from multiprocessing import cpu_count, Manager
import concurrent.futures

from numba import njit

from config import config
from models.route_manager import RouteManager
from utils.early_stopper import EarlyStopper
from solvers.allocate_ga import StoreAllocationGA

@njit(cache=True)
def _njit_weights(ids_array, dist_matrix, volumes):
    w = np.zeros(len(ids_array) - 2)
    for i in range(1, len(ids_array) - 1):
        d = (dist_matrix[ids_array[i-1], ids_array[i]]
             + dist_matrix[ids_array[i], ids_array[i+1]]
             - dist_matrix[ids_array[i-1], ids_array[i+1]])
        w[i-1] = max(d, 1e-6) * volumes[i-1]
    return w / np.sum(w)

# ── Worker globals ──────────────────────────────────────────────────────────
ROUTES = None
DIST   = None
TIME   = None

def init_worker(main_routes, distance_matrix, time_matrix):
    global ROUTES, DIST, TIME
    ROUTES = main_routes
    DIST   = distance_matrix
    TIME   = time_matrix

def fitness_worker(args):
    """Evaluate a full decoded context ({route_id: [store_dicts]})."""
    context_decoded, cache_key = args

    copy_routes = {}
    store_list = []
    
    for r_id, rd in ROUTES.items():
        if r_id in context_decoded:
            removed_ids = {s['route_code'] for s in context_decoded[r_id]}
            kept_stores = [s for s in rd["stores"] if s['route_code'] not in removed_ids]
            store_list.extend(context_decoded[r_id])
            
            new_volume = sum(s.get('volume', 0) for s in kept_stores)
            new_dc = rd["dc"].copy()
            new_dc['total_volume'] = new_volume
            new_dc['load_rate'] = new_volume / new_dc.get('max_capacity', 1)
            
            copy_routes[r_id] = {"dc": new_dc, "stores": kept_stores}
        else:
            copy_routes[r_id] = {"dc": rd["dc"], "stores": rd["stores"]}

    ac_cost, _, _, ac_vn = StoreAllocationGA(
        copy_routes, store_list, DIST, TIME,
        population_size=0, generations=0
    ).run(return_routes=False)

    result = {'cost': ac_cost, 'vn': ac_vn}
    return cache_key, result


# ── CCEA class ───────────────────────────────────────────────────────────────
class StoreExtractionGA:
    """
    Cooperative Co-Evolutionary GA with binary encoding.

    Each overloaded route is one subcomponent.
    Subpopulation P_i contains binary vectors over the stores of route i.
    bit j = 1  →  extract store j   (remove from main route)
    bit j = 0  →  keep store j
    """

    def __init__(self, main_routes, distance_matrix, time_matrix,
                 population_size=10, elite_rate=0.1, generations=50,
                 cross_rate=0.8, mutation_rate=0.2, early_stop_patience=100):
        self.dc                  = config.DC_CONFIG
        self.distance_matrix     = distance_matrix
        self.time_matrix         = time_matrix
        self.main_routes         = self._init_routes(main_routes)
        self.population_size     = population_size
        self.elite_size          = max(1, int(elite_rate * population_size))
        self.generations         = generations
        self.cross_rate          = cross_rate
        self.mutation_rate       = mutation_rate
        self.early_stop_patience = early_stop_patience
        self.overloaded_routes   = self._get_overloaded()
        # Fixed store order per route (index → store dict)
        self.route_stores        = {r: list(info['stores'])
                                    for r, info in self.main_routes.items()}
        self.best_cost           = float('inf')
        self.best_individual     = None   # {route_id: binary_vector}
        self.log                 = []
        
        # Numba arrays mapping
        self.s2i = {s: i for i, s in enumerate(self.distance_matrix.keys())}
        n = len(self.s2i)
        self.np_dist = np.zeros((n, n))
        for s1, i1 in self.s2i.items():
            for s2, i2 in self.s2i.items():
                if s1 in self.distance_matrix and s2 in self.distance_matrix[s1]:
                    self.np_dist[i1, i2] = self.distance_matrix[s1][s2]

    # ── Route helpers ─────────────────────────────────────────────────────────

    def _init_routes(self, routes):
        RouteManager(routes, self.distance_matrix, self.time_matrix).update_all_routes_info()
        return routes

    def _copy_routes(self, routes):
        return {r: {"dc": d["dc"].copy(), "stores": [s.copy() for s in d["stores"]]}
                for r, d in routes.items()}

    def _get_overloaded(self):
        return {r: info for r, info in self.main_routes.items()
                if info['dc']['load_rate'] > 1.0}

    # ── Binary encoding helpers ───────────────────────────────────────────────

    def _decode(self, route_id, bits):
        """Binary vector → list of extracted store dicts."""
        return [self.route_stores[route_id][j]
                for j, b in enumerate(bits) if b == 1]

    def _decode_context(self, context):
        """Binary context → {route_id: [store_dicts]}."""
        return {r: self._decode(r, bits) for r, bits in context.items()}


    # ── Heuristic removal weights ─────────────────────────────────────────────

    def _weights(self, stores):
        depot_id = self.s2i[self.dc['store_id']]
        ids_array = np.array([depot_id] + [self.s2i[s['store_id']] for s in stores] + [depot_id], dtype=np.int64)
        volumes = np.array([s['volume'] for s in stores], dtype=np.float64)
        return _njit_weights(ids_array, self.np_dist, volumes)

    # ── Feasibility repair (binary) ───────────────────────────────────────────

    def _repair_binary(self, route_id, bits):
        """
        Ensure the binary vector yields a capacity-feasible remaining route.
        Uses heuristic weights (detour cost × volume) to select which stores
        to extract (flip 0 → 1) until the route is no longer overloaded.
        """
        bits       = list(bits)
        all_stores = self.route_stores[route_id]
        capacity   = self.main_routes[route_id]['dc']['max_capacity']

        while True:
            rem  = [all_stores[j] for j, b in enumerate(bits) if b == 0]
            load = sum(s['volume'] for s in rem)
            if load <= capacity:
                break
            probs  = self._weights(rem)
            chosen = np.random.choice(len(rem), p=probs)
            global_j = next(j for j, s in enumerate(all_stores)
                            if s['store_id'] == rem[chosen]['store_id'])
            bits[global_j] = 1

        return bits

    # ── Individual initialisation ─────────────────────────────────────────────

    def _init_binary(self, route_id):
        """Generate one feasible binary individual for route_id."""
        n    = len(self.route_stores[route_id])
        bits = [0] * n
        return self._repair_binary(route_id, bits)

    # ── Subpopulation init ────────────────────────────────────────────────────

    def _init_subpops(self):
        return {r: [self._init_binary(r) for _ in range(self.population_size)]
                for r in self.overloaded_routes}

    # ── Encoding (hash) ───────────────────────────────────────────────────────

    def _encode_ctx(self, ctx):
        return hash(tuple((k, tuple(ctx[k])) for k in sorted(ctx.keys())))

    # ── Crossover (binary uniform + repair) ──────────────────────────────────

    def _crossover_sub(self, route_id, p1, p2):
        """Uniform binary crossover, followed by feasibility repair."""
        if random.random() >= self.cross_rate:
            return list(p1)
        child = [p1[j] if random.random() < 0.5 else p2[j]
                 for j in range(len(p1))]
        return self._repair_binary(route_id, child)

    # ── Mutation (bit flip + repair) ──────────────────────────────────────────

    def _mutate_sub(self, route_id, bits):
        """Bit-flip mutation: each bit independently flips with probability mutation_rate.
        Repair is applied once after all flips to restore capacity feasibility."""
        bits   = list(bits)
        flipped = False
        for j in range(len(bits)):
            if random.random() < self.mutation_rate:
                bits[j] = 1 - bits[j]
                flipped = True
        return self._repair_binary(route_id, bits) if flipped else bits

    # ── Parent selection ──────────────────────────────────────────────────────

    def _select(self, subpop, costs):
        inv = np.array([1.0 / max(c, 1e-12) for c in costs])
        s   = inv.sum()
        prob = inv / s if s > 0 and np.isfinite(s) else np.ones(len(subpop)) / len(subpop)
        i, j = np.random.choice(len(subpop), size=2, p=prob)
        return subpop[i], subpop[j]

    # ── Route reconstruction ──────────────────────────────────────────────────

    def _best_main_routes(self, individual):
        rm = RouteManager(self._copy_routes(self.main_routes),
                          self.distance_matrix, self.time_matrix)
        for r_id, bits in individual.items():
            for s in self._decode(r_id, bits):
                rm.remove_store(r_id, s)
        return rm.routes_info

    def _individual_to_list(self, individual):
        return [s for r_id, bits in individual.items()
                for s in self._decode(r_id, bits)]

    def run(self):
        route_ids = list(self.overloaded_routes.keys())
        N = self.population_size

        early_stopper = EarlyStopper(patience=self.early_stop_patience)
        local_cache = {}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, cpu_count() - 2),
            initializer=init_worker,
            initargs=(self.main_routes, self.distance_matrix, self.time_matrix)
        ) as pool:

            # ── Step 2: Randomly initialize and evaluate subpopulations and context vector b ──
            subpops = self._init_subpops()
            
            # Randomly initialize b and set fb = f(b)
            context = {r: list(random.choice(subpops[r])) for r in route_ids}
            ctx_key = self._encode_ctx(context)
            if ctx_key not in local_cache:
                for k, res in pool.map(fitness_worker, [(self._decode_context(context), ctx_key)]):
                    local_cache[k] = res
            
            fb = local_cache[ctx_key]['cost']
            self.best_cost = fb
            self.best_individual = {k: list(v) for k, v in context.items()}

            # Initialize subpopulations Pi and evaluate
            subpop_costs = {r: [float('inf')] * N for r in route_ids}
            
            eval_batch = []
            seen_keys = set()
            for r in route_ids:
                for bits in subpops[r]:
                    ctx = {k: list(v) for k, v in context.items()}; ctx[r] = bits
                    key = self._encode_ctx(ctx)
                    if key not in local_cache and key not in seen_keys:
                        eval_batch.append((self._decode_context(ctx), key))
                        seen_keys.add(key)
                        
            if eval_batch:
                chunk_size = max(1, len(eval_batch) // max(1, cpu_count() - 2))
                for k, res in pool.map(fitness_worker, eval_batch, chunksize=chunk_size):
                    local_cache[k] = res
                    
            for r in route_ids:
                for j, bits in enumerate(subpops[r]):
                    ctx = {k: list(v) for k, v in context.items()}; ctx[r] = bits
                    key = self._encode_ctx(ctx)
                    subpop_costs[r][j] = local_cache.get(key, {'cost': float('inf')})['cost']
                    # Update context if initial subpop individual is better
                    if subpop_costs[r][j] < fb:
                        fb = subpop_costs[r][j]
                        context[r] = bits
                        self.best_cost = fb
                        self.best_individual = {k: list(v) for k, v in context.items()}

            # ── Step 3: Evolution ──
            for gen_idx in range(self.generations):
                for route_id in route_ids:
                    Pi = subpops[route_id]
                    prev_costs = subpop_costs[route_id]
                    
                    # 3.1 Generate an offspring subpopulation Oi
                    Oi = []
                    for _ in range(N):
                        p1, p2 = self._select(Pi, prev_costs)
                        child  = self._crossover_sub(route_id, p1, p2)
                        child  = self._mutate_sub(route_id, child)
                        Oi.append(child)
                        
                    # 3.2 Individual fitness assignment for Oi
                    eval_batch = []
                    Oi_keys = []
                    seen_keys = set()
                    for child in Oi:
                        ctx = {k: list(v) for k, v in context.items()}; ctx[route_id] = child
                        key = self._encode_ctx(ctx)
                        Oi_keys.append(key)
                        if key not in local_cache and key not in seen_keys:
                            eval_batch.append((self._decode_context(ctx), key))
                            seen_keys.add(key)
                            
                    if eval_batch:
                        chunk_size = max(1, len(eval_batch) // max(1, cpu_count() - 2))
                        for k, res in pool.map(fitness_worker, eval_batch, chunksize=chunk_size):
                            local_cache[k] = res
                            
                    Oi_costs = []
                    for key in Oi_keys:
                        Oi_costs.append(local_cache.get(key, {'cost': float('inf')})['cost'])
                        
                    # 3.3 Update the context vector b
                    for j, child in enumerate(Oi):
                        if Oi_costs[j] < fb:
                            context[route_id] = child
                            fb = Oi_costs[j]
                            self.best_cost = fb
                            self.best_individual = {k: list(v) for k, v in context.items()}
                            
                    # 3.4 Select the next parent subpopulation Pi <- Select(Pi, Oi)
                    all_bits = Pi + Oi
                    all_costs = prev_costs + Oi_costs
                    sorted_i = np.argsort(all_costs)
                    subpops[route_id] = [all_bits[i] for i in sorted_i[:N]]
                    subpop_costs[route_id] = [all_costs[i] for i in sorted_i[:N]]

                # Logging
                ctx_key = self._encode_ctx(context)
                res = local_cache.get(ctx_key, {'cost': fb, 'vn': 0})
                vn = res['vn']
                print(f'Store Extraction: iteration{gen_idx + 1} -> '
                      f'vn = {vn}, cost = {fb - vn * 2000:.2f}, fitness = {fb:.2f}')

                self.log.append({
                    'generation': gen_idx + 1,
                    'iter_best_cost': fb,
                    'best_cost': fb,
                })

                if early_stopper.check(fb):
                    break

        if self.best_individual is None:
            self.best_individual = context

        return (self._best_main_routes(self.best_individual),
                self._individual_to_list(self.best_individual))