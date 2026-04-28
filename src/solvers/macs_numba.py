import time
import random
import numpy as np
from numba import njit
from datetime import datetime, timedelta

from config.config import DC_CONFIG
from solvers.vnd import VND
from utils.early_stopper import EarlyStopper

# ─────────────────────────────────────────────────────────────────────────────
# 1. Numba 核心加速區 (嚴格還原 MACS-VRPTW 邏輯)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _njit_eval_route(route, np_dist, np_time, np_demand, np_ready, np_due, np_service, capacity, depot_due, is_solomon):
    """嚴格還原原版 _eval_route"""
    load = 0.0
    cur_time = np_ready[0]
    total_dist = 0.0
    prev = 0
    
    for i in range(len(route)):
        idx = route[i]
        travel_time = np_time[prev, idx]
        travel_dist = np_dist[prev, idx]
        arrival = cur_time + travel_time
        
        if arrival > np_due[idx] + 1e-6:
            return False, float('inf'), load

        if is_solomon:
            start = max(arrival, np_ready[idx])
        else:
            if arrival < np_ready[idx] - 1e-6:
                return False, float('inf'), load
            start = arrival

        cur_time = start + np_service[idx]
        total_dist += travel_dist
        load += np_demand[idx]
        
        if load > capacity + 1e-6:
            return False, float('inf'), load
        prev = idx
    
    # 回到 Depot
    total_dist += np_dist[prev, 0]
    if cur_time + np_time[prev, 0] > depot_due + 1e-6:
        return False, float('inf'), load
        
    return True, total_dist, load

@njit(cache=True)
def _njit_choose_next(cur, cand_arr, ph, np_dist, np_time, np_ready, np_due, in_vec, cur_time, beta, q0, rand_val, is_solomon):
    """嚴格還原原版 ETA 公式與 IN 陣列機制"""
    scores = np.zeros(len(cand_arr), dtype=np.float64)
    best_idx = -1
    best_score = -1.0
    
    for i in range(len(cand_arr)):
        j = cand_arr[i]
        travel_time = np_time[cur, j]
        arrival = cur_time + travel_time
        
        if is_solomon:
            delivery = max(arrival, np_ready[j])
        else:
            delivery = arrival
            
        delta = delivery - cur_time
        # 原版公式: d = delta * max(1.0, (due_date - cur_time))
        d = delta * max(1.0, (np_due[j] - cur_time))
        d = max(1.0, d - in_vec[j]) # 套用未探訪懲罰/獎勵
        eta = 1.0 / d
        
        score = ph[cur, j] * (eta ** beta)
        scores[i] = score
        
        if score > best_score:
            best_score = score
            best_idx = j
            
    if rand_val < q0:
        return best_idx
        
    total = np.sum(scores)
    if total < 1e-12:
        return cand_arr[int(rand_val * len(cand_arr))]
        
    r = rand_val * total
    cum = 0.0
    for i in range(len(cand_arr)):
        cum += scores[i]
        if cum >= r: return cand_arr[i]
    return cand_arr[-1]

@njit
def _njit_build_ant(num_vehicles, n_nodes, ph, np_dist, np_time, np_demand, 
                    np_ready, np_due, np_service, capacity, depot_due, 
                    in_vec, beta, rho, q0, tau0, is_solomon, rand_vals):
    """使用 Numba 加速建構單隻螞蟻的解"""
    unvisited = np.ones(n_nodes, dtype=np.bool_)
    unvisited[0] = False
    unvisited_count = n_nodes - 1
    
    routes_flat = np.zeros(n_nodes * 2, dtype=np.int64)
    routes_len = np.zeros(num_vehicles, dtype=np.int64)
    
    ptr = 0
    depots_left = num_vehicles
    v_idx = 0
    
    cur = 0
    cur_load = 0.0
    cur_time = np_ready[0]
    
    feasible_buf = np.zeros(n_nodes, dtype=np.int64)
    r_val_idx = 0
    
    while unvisited_count > 0 and depots_left > 0:
        f_count = 0
        for cid in range(1, n_nodes):
            if not unvisited[cid]: continue
            if cur_load + np_demand[cid] > capacity + 1e-6: continue
            
            travel_time = np_time[cur, cid]
            arrival = cur_time + travel_time
            if arrival > np_due[cid] + 1e-6: continue
            
            if is_solomon:
                start = max(arrival, np_ready[cid])
            else:
                if arrival < np_ready[cid] - 1e-6: continue
                start = arrival
                
            if start + np_service[cid] + np_time[cid, 0] > depot_due + 1e-6: continue
            
            feasible_buf[f_count] = cid
            f_count += 1
            
        if f_count == 0:
            if routes_len[v_idx] > 0:
                v_idx += 1
            cur, cur_load, cur_time = 0, 0.0, np_ready[0]
            depots_left -= 1
            continue
            
        cand_arr = feasible_buf[:f_count]
        r_val = rand_vals[r_val_idx % len(rand_vals)]
        r_val_idx += 1
        
        chosen = _njit_choose_next(cur, cand_arr, ph, np_dist, np_time, np_ready, np_due, in_vec, cur_time, beta, q0, r_val, is_solomon)
        
        # Local Update
        ph[cur, chosen] = (1.0 - rho) * ph[cur, chosen] + rho * tau0
        ph[chosen, cur] = (1.0 - rho) * ph[chosen, cur] + rho * tau0
        
        travel_time = np_time[cur, chosen]
        arrival = cur_time + travel_time
        start = max(arrival, np_ready[chosen]) if is_solomon else arrival
        cur_time = start + np_service[chosen]
        cur_load += np_demand[chosen]
        
        routes_flat[ptr] = chosen
        ptr += 1
        routes_len[v_idx] += 1
        unvisited[chosen] = False
        unvisited_count -= 1
        cur = chosen
        
    # 🟢 修正處：加上 v_idx < num_vehicles 的安全邊界檢查
    if v_idx < num_vehicles and routes_len[v_idx] > 0:
        v_idx += 1
        
    return routes_flat, routes_len, v_idx, unvisited

@njit
def _njit_global_update(ph, routes_flat, routes_len, v_count, cost, rho):
    if cost <= 0 or cost >= 1e10: return
    inc = rho * (1.0 / cost)
    ptr = 0
    for v in range(v_count):
        l = routes_len[v]
        if l == 0: continue
        route = routes_flat[ptr:ptr+l]
        ptr += l
        
        a, b = 0, route[0]
        ph[a, b] = (1.0 - rho) * ph[a, b] + inc
        ph[b, a] = (1.0 - rho) * ph[b, a] + inc
        
        for i in range(l - 1):
            a, b = route[i], route[i+1]
            ph[a, b] = (1.0 - rho) * ph[a, b] + inc
            ph[b, a] = (1.0 - rho) * ph[b, a] + inc
            
        a, b = route[-1], 0
        ph[a, b] = (1.0 - rho) * ph[a, b] + inc
        ph[b, a] = (1.0 - rho) * ph[b, a] + inc


# ─────────────────────────────────────────────────────────────────────────────
# 2. 混合 Python 層 (處理複雜物件如 VND、Insertion 與資料轉換)
# ─────────────────────────────────────────────────────────────────────────────

class MACSNumbaSolver:
    def __init__(self, remaining_stores, distance_matrix, time_matrix,
                 num_ants=10, iterations=100, support_capacity=200,
                 time_limit_per_route=1236.0, is_solomon=False,
                 beta=1.0, rho=0.1, q0=0.9, early_stop_patience=10,
                 verbose=True, vnd_strategy='best'):

        self.remaining_stores = remaining_stores
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.verbose = verbose
        self.iterations = iterations
        self.is_solomon = is_solomon
        self.early_stop_patience = early_stop_patience
        self.vnd_strategy = vnd_strategy
        self.capacity = float(support_capacity)
        self.depot_due = float(time_limit_per_route)
        
        self.num_ants = max(num_ants, 5)
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        
        self.vnd = VND(distance_matrix, time_matrix, vehicle_cost=0, is_solomon=is_solomon, 
                       vnd_strategy=self.vnd_strategy, time_limit=time_limit_per_route)
        self.dc_config = DC_CONFIG.copy()
        self.dc_config['max_capacity'] = support_capacity

        self._prepare_numpy_data()
        self.n_nodes = len(self.np_demand)
        self.tau0 = 1.0 / (self.n_nodes * 10.0)
        
        self.ph_vei = np.full((self.n_nodes, self.n_nodes), self.tau0, dtype=np.float64)
        self.ph_time = np.full((self.n_nodes, self.n_nodes), self.tau0, dtype=np.float64)

    def _prepare_numpy_data(self):
        """將資料轉為 Numba 友善的 Numpy 陣列 (統一為分鐘數)"""
        n = len(self.remaining_stores) + 1
        self.np_demand = np.zeros(n, dtype=np.float64)
        self.np_ready = np.zeros(n, dtype=np.float64)
        self.np_due = np.zeros(n, dtype=np.float64)
        self.np_service = np.zeros(n, dtype=np.float64)
        
        self.stores_internal = []
        base_dt = datetime(2024, 1, 1, 0, 0, 0)
        
        # Depot
        self.np_due[0] = self.depot_due
        self.stores_internal.append({'sid': 'dc', 'idx': 0})
        
        store_ids = sorted(self.remaining_stores, key=lambda s: int(s.get('store_id', 0)) if str(s.get('store_id')).isdigit() else str(s.get('store_id')))
        
        for i, s in enumerate(store_ids, start=1):
            rt = datetime.fromisoformat(s['earliest_time'])
            dt = datetime.fromisoformat(s['latest_time'])
            
            self.np_demand[i] = float(s['volume'])
            self.np_ready[i] = (rt - base_dt).total_seconds() / 60.0
            self.np_due[i] = (dt - base_dt).total_seconds() / 60.0
            self.np_service[i] = float(s['dwell_time'])
            self.stores_internal.append({'sid': s['store_id'], 'idx': i, 'store_info': s})
            
        self.np_dist = np.zeros((n, n), dtype=np.float64)
        self.np_time = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    si, sj = self.stores_internal[i]['sid'], self.stores_internal[j]['sid']
                    self.np_dist[i, j] = float(self.distance_matrix.get(si, {}).get(sj, 0.0))
                    self.np_time[i, j] = float(self.time_matrix.get(si, {}).get(sj, 0.0))

    def _insertion_python(self, routes, unvisited_bool_arr):
        """保留原版的 Greedy Insertion 以拯救漏掉的點"""
        unvisited = [i for i, unv in enumerate(unvisited_bool_arr) if unv and i > 0]
        unvisited.sort(key=lambda x: -self.np_demand[x])
        
        for cid in unvisited:
            best_cost = float('inf')
            best_r, best_p = None, None
            for ri, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_r = route[:pos] + [cid] + route[pos:]
                    # Numpy eval is fast enough here
                    f, cost, _ = _njit_eval_route(
                        np.array(new_r, dtype=np.int64), self.np_dist, self.np_time, 
                        self.np_demand, self.np_ready, self.np_due, self.np_service, 
                        self.capacity, self.depot_due, self.is_solomon
                    )
                    if f and cost < best_cost:
                        best_cost = cost
                        best_r, best_p = ri, pos
            if best_r is not None:
                routes[best_r].insert(best_p, cid)
        return routes

    def _apply_vnd(self, routes):
        routes_info = {}
        for v_id, route in enumerate(routes):
            r_id = f"9{v_id:02d}"
            stores = [self.stores_internal[idx]['store_info'].copy() for idx in route]
            dc_copy = self.dc_config.copy()
            dc_copy['route_id'] = r_id
            dc_copy['route_code'] = r_id
            for i, s in enumerate(stores):
                s['route_id'] = r_id
                s['route_code'] = f"{r_id}{i:02d}"
                
            routes_info[r_id] = {'dc': dc_copy, 'stores': stores}
        
        opt_routes_info, _ = self.vnd.optimize(routes_info)
        sid_to_idx = {s['sid']: s['idx'] for s in self.stores_internal[1:]}
        new_routes = []
        for k, v in opt_routes_info.items():
            if not v['stores']: continue
            new_routes.append([sid_to_idx[s['store_id']] for s in v['stores']])
        return new_routes

    def run(self):
        t_start = time.time()
        
        # 1. 產生初始解 (Nearest Neighbor - 在 Numba 中可以快速跑一次, 但這裡沿用簡單啟發)
        gb_routes = []
        unvisited = set(range(1, self.n_nodes))
        while unvisited:
            route, cur, cur_time, load = [], 0, self.np_ready[0], 0.0
            while True:
                best_dist, best = float('inf'), None
                for cid in unvisited:
                    if load + self.np_demand[cid] > self.capacity: continue
                    arr = cur_time + self.np_time[cur, cid]
                    if arr > self.np_due[cid]: continue
                    start = max(arr, self.np_ready[cid]) if self.is_solomon else (arr if arr >= self.np_ready[cid] else -1)
                    if start == -1 or start + self.np_service[cid] + self.np_time[cid, 0] > self.depot_due: continue
                    
                    if self.np_dist[cur, cid] < best_dist:
                        best_dist, best = self.np_dist[cur, cid], cid
                if best is None: break
                
                arr = cur_time + self.np_time[cur, best]
                start = max(arr, self.np_ready[best]) if self.is_solomon else arr
                cur_time = start + self.np_service[best]
                load += self.np_demand[best]
                route.append(best)
                unvisited.remove(best)
                cur = best
            if route: gb_routes.append(route)
            elif unvisited:
                cid = unvisited.pop()
                gb_routes.append([cid])

        # Insertion to rescue unvisited stores (mirrors macs.py)
        all_visited = set(n for r in gb_routes for n in r)
        unvisited_bool = np.array([i not in all_visited for i in range(self.n_nodes)], dtype=np.bool_)
        unvisited_bool[0] = False
        gb_routes = self._insertion_python(gb_routes, unvisited_bool)

        # Initial Cost
        gb_cost = sum(_njit_eval_route(np.array(r, dtype=np.int64), self.np_dist, self.np_time, self.np_demand, self.np_ready, self.np_due, self.np_service, self.capacity, self.depot_due, self.is_solomon)[1] for r in gb_routes)
        gb_nv = len(gb_routes)
        
        if self.verbose:
            print(f"    [MACS-Accel] Init: {gb_nv} vehicles, dist={gb_cost:.2f}")

        early_stopper = EarlyStopper(patience=self.early_stop_patience)
        in_vec = np.zeros(self.n_nodes, dtype=np.float64)

        for iter_num in range(self.iterations):
            v = gb_nv
            
            # --- VEI Cycle (Minimizing Vehicles, mirrors macs.py inner_max=10) ---
            found_fewer = False
            inner_max = 20
            for _inner in range(inner_max):
                vei_feasible = None
                best_vei_visited = 0
                
                for _ in range(self.num_ants):
                    r_vals = np.random.rand(self.n_nodes * 2)
                    r_flat, r_len, v_cnt, unv = _njit_build_ant(
                        max(1, v - 1), self.n_nodes, self.ph_vei, self.np_dist, self.np_time, 
                        self.np_demand, self.np_ready, self.np_due, self.np_service, 
                        self.capacity, self.depot_due, in_vec, self.beta, self.rho, self.q0, 
                        self.tau0, self.is_solomon, r_vals
                    )
                    
                    # Reconstruct Python list for Insertion
                    routes = []
                    ptr = 0
                    for i in range(v_cnt):
                        l = r_len[i]
                        if l > 0: routes.append(list(r_flat[ptr:ptr+l]))
                        ptr += l
                    
                    routes = self._insertion_python(routes, unv)
                    
                    # 計算是否完全走完
                    visited_cnt = sum(len(r) for r in routes)
                    
                    # VEI 核心：紀錄未探訪點
                    if visited_cnt < self.n_nodes - 1:
                        visited_set = set(item for sublist in routes for item in sublist)
                        for cid in range(1, self.n_nodes):
                            if cid not in visited_set:
                                in_vec[cid] += 1.0
                                
                    if visited_cnt > best_vei_visited:
                        best_vei_visited = visited_cnt
                        all_f = True
                        cost = 0
                        for r in routes:
                            f, d, _ = _njit_eval_route(np.array(r, dtype=np.int64), self.np_dist, self.np_time, self.np_demand, self.np_ready, self.np_due, self.np_service, self.capacity, self.depot_due, self.is_solomon)
                            if not f: all_f = False
                            cost += d
                        if all_f and visited_cnt == self.n_nodes - 1:
                            vei_feasible = (routes, cost)
                            in_vec.fill(0.0)  # 找到可行解重置 IN
                
                if vei_feasible:
                    vr, vc = vei_feasible
                    _flat = np.array([n for r in vr for n in r], dtype=np.int64)
                    _len = np.array([len(r) for r in vr], dtype=np.int64)
                    _njit_global_update(self.ph_vei, _flat, _len, len(vr), vc, self.rho)
                    if len(vr) < gb_nv:
                        gb_routes, gb_cost, gb_nv = vr, vc, len(vr)
                        v = gb_nv  # 更新 v 讓下一輪 VEI 嘗試更少車輛
                        found_fewer = True
                        break  # 成功減少車輛，跳出 inner loop
                    elif len(vr) == gb_nv and vc < gb_cost:
                        gb_routes, gb_cost = vr, vc

            # Global update VEI with GB
            _flat = np.array([n for r in gb_routes for n in r], dtype=np.int64)
            _len = np.array([len(r) for r in gb_routes], dtype=np.int64)
            _njit_global_update(self.ph_vei, _flat, _len, len(gb_routes), gb_cost, self.rho)

            # --- TIME Cycle (Minimizing Distance) ---
            best_time_sol, best_time_cost = None, float('inf')
            dummy_in = np.zeros(self.n_nodes, dtype=np.float64)
            for _ in range(self.num_ants):
                r_vals = np.random.rand(self.n_nodes * 2)
                r_flat, r_len, v_cnt, unv = _njit_build_ant(
                    gb_nv, self.n_nodes, self.ph_time, self.np_dist, self.np_time, 
                    self.np_demand, self.np_ready, self.np_due, self.np_service, 
                    self.capacity, self.depot_due, dummy_in, self.beta, self.rho, self.q0, 
                    self.tau0, self.is_solomon, r_vals
                )
                
                routes = []
                ptr = 0
                for i in range(v_cnt):
                    l = r_len[i]
                    if l > 0: routes.append(list(r_flat[ptr:ptr+l]))
                    ptr += l
                
                routes = self._insertion_python(routes, unv)
                all_f = True
                cost = 0
                for r in routes:
                    f, d, _ = _njit_eval_route(np.array(r, dtype=np.int64), self.np_dist, self.np_time, self.np_demand, self.np_ready, self.np_due, self.np_service, self.capacity, self.depot_due, self.is_solomon)
                    if not f: all_f = False
                    cost += d
                    
                if all_f and cost < best_time_cost and sum(len(r) for r in routes) == self.n_nodes - 1:
                    best_time_cost = cost
                    best_time_sol = [r[:] for r in routes]
                    
            if best_time_sol:
                # VND Optimization
                opt_routes = self._apply_vnd(best_time_sol)
                f_opt = True
                cost_opt = 0
                for r in opt_routes:
                    f, d, _ = _njit_eval_route(np.array(r, dtype=np.int64), self.np_dist, self.np_time, self.np_demand, self.np_ready, self.np_due, self.np_service, self.capacity, self.depot_due, self.is_solomon)
                    if not f: f_opt = False
                    cost_opt += d
                if f_opt and cost_opt < best_time_cost:
                    best_time_cost = cost_opt
                    best_time_sol = opt_routes

                if best_time_cost < gb_cost:
                    gb_routes, gb_cost = best_time_sol, best_time_cost
                    gb_nv = len(gb_routes)

            # Global update TIME with GB
            _flat = np.array([n for r in gb_routes for n in r], dtype=np.int64)
            _len = np.array([len(r) for r in gb_routes], dtype=np.int64)
            _njit_global_update(self.ph_time, _flat, _len, len(gb_routes), gb_cost, self.rho)

            if early_stopper.check((gb_nv, gb_cost)):
                if self.verbose: print(f"    [MACS-Accel] Early stop triggered at iter {iter_num}.")
                break

            if self.verbose:
                elapsed = time.time() - t_start
                print(f"    [MACS-Accel Iter {iter_num}] Time: {elapsed:.1f}s, V: {gb_nv}, Dist: {gb_cost:.2f}")

        elapsed = time.time() - t_start
        if self.verbose:
            print(f"    [MACS-Accel] Done in {elapsed:.1f}s → {gb_nv} vehicles, dist={gb_cost:.2f}")

        return (gb_nv, gb_cost), self._format_solution(gb_routes)

    def _format_solution(self, routes):
        """將最終路徑格式化回前端或後續需要的 JSON dict 格式"""
        base_dt = datetime(2024, 1, 1, 0, 0, 0)
        res = {}
        for v_id, route in enumerate(routes):
            r_id = f"V{v_id+1:02d}"
            total_dist, total_load, cur_time, prev = 0.0, 0.0, self.np_ready[0], 0
            formatted_stores = []
            
            for i, idx in enumerate(route):
                s_info = self.stores_internal[idx]['store_info'].copy()
                travel_dist = self.np_dist[prev, idx]
                travel_time = self.np_time[prev, idx]
                arrival = cur_time + travel_time
                
                if self.is_solomon: start = max(arrival, self.np_ready[idx])
                else: start = arrival
                    
                s_info['pred_time'] = (base_dt + timedelta(minutes=start)).isoformat(timespec='seconds')
                s_info['route_id'] = r_id
                s_info['route_code'] = f"{r_id}{i+1:02d}"
                formatted_stores.append(s_info)
                
                cur_time = start + self.np_service[idx]
                total_dist += travel_dist
                total_load += self.np_demand[idx]
                prev = idx
                
            total_dist += self.np_dist[prev, 0]
            total_time = cur_time + self.np_time[prev, 0]
            
            res[r_id] = {
                "dc": {
                    "route_id": r_id, "route_code": r_id,
                    "store_id": self.dc_config.get('store_id', 'dc'),
                    "store_name": self.dc_config.get('store_name', 'Solomon Depot'),
                    "total_volume": float(total_load),
                    "load_rate": float(total_load / self.capacity) if self.capacity > 0 else 0.0,
                    "max_capacity": float(self.capacity),
                    "region": self.dc_config.get('region', 'unknown'),
                    "distance": float(total_dist), "duration": float(total_time)
                },
                "stores": formatted_stores
            }
        return res