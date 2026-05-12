import time
import random
import numpy as np
import threading
import queue
from numba import njit
from datetime import datetime, timedelta

from config.config import DC_CONFIG
from solvers.vnd import VND
from utils.early_stopper import EarlyStopper

# ─────────────────────────────────────────────────────────────────────────────
# 1. Numba Core
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _njit_dist_heuristic(current_idx, next_idx, distance_matrix):
    if current_idx == next_idx:
        return 0.0
    return 1.0 / (distance_matrix[current_idx, next_idx] + 1e-12)

@njit(cache=True)
def _njit_eval_route(route, nodes_idx, dc_idx, np_dist, np_time, np_volume, np_dwell, np_earliest, np_latest, np_sched, capacity, time_limit, is_solomon, dc_departure_time):
    load = 0.0
    cur_time = dc_departure_time if not is_solomon else 0.0
    total_dist = 0.0
    curr_duration = 0.0
    prev = dc_idx
    
    for i in range(len(route)):
        idx = route[i]
        travel_time = np_time[prev, idx]
        travel_dist = np_dist[prev, idx]
        
        prev_dwell = np_dwell[prev] if prev != dc_idx else 0
        if not is_solomon and prev == dc_idx:
            arrival = np_sched[idx]
        else:
            arrival = cur_time + travel_time + prev_dwell
            
        if arrival > np_latest[idx]:
            return False, 1e12, load
            
        if is_solomon:
            start = max(arrival, np_earliest[idx])
            pre_to_dc = np_time[prev, dc_idx]
            cur_to_dc = np_time[idx, dc_idx]
            curr_duration = curr_duration + (travel_time + cur_to_dc - pre_to_dc) + (start - arrival) + np_dwell[idx]
        else:
            start = arrival
            if prev == dc_idx:
                curr_duration = np_time[dc_idx, idx] + np_dwell[idx] + np_time[idx, dc_idx]
            else:
                pre_to_dc = np_time[prev, dc_idx]
                cur_to_dc = np_time[idx, dc_idx]
                curr_duration = curr_duration + (travel_time + cur_to_dc - pre_to_dc) + np_dwell[idx]
                
        if curr_duration > time_limit:
            return False, 1e12, load
            
        cur_time = start + np_dwell[idx]
        total_dist += travel_dist
        load += np_volume[idx]
        if load > capacity:
            return False, 1e12, load
        prev = idx
        
    total_dist += np_dist[prev, dc_idx]
    return True, total_dist, load

@njit(cache=True)
def _njit_nearest_neighbor(n_nodes, np_dist, np_time, np_volume, np_dwell, np_earliest, np_latest, np_sched, np_group, np_region, dc_departure_time, capacity, time_limit, is_solomon):
    unvisited = np.ones(n_nodes, dtype=np.bool_)
    unvisited[0] = False
    unvisited_count = n_nodes - 1
    
    routes_flat = np.zeros(n_nodes * 2, dtype=np.int64)
    routes_len = np.zeros(n_nodes, dtype=np.int64)
    ptr = 0
    v_idx = 0
    
    while unvisited_count > 0:
        cur = 0
        cur_vol = 0.0
        cur_duration = 0.0
        prev_pred_time_epoch = dc_departure_time if not is_solomon else 0
        
        route_found = False
        while unvisited_count > 0:
            best_dist = 1e12
            best_idx = -1
            
            # Find nearest feasible
            for cid in range(1, n_nodes):
                if not unvisited[cid]: continue
                
                # Capacity
                if cur_vol + np_volume[cid] > capacity: continue
                
                # Region constraint (simplified for NN)
                if not is_solomon:
                    last_g, last_r = np_group[cur], np_region[cur]
                    store_g, store_r = np_group[cid], np_region[cid]
                    if last_g == 2:
                        if (last_r == 0 and store_r == 1) or (last_r == 1 and store_r == 0) or (last_r == 2 and store_r == 3) or (last_r == 3 and store_r == 2): continue
                    if last_g == 2 and store_g not in (0, 1, 2): continue
                    elif last_g == 1 and store_g not in (0, 1): continue
                    elif last_g == 0 and store_g != 0: continue
                
                # Time
                pre_to_cur = np_time[cur, cid]
                prev_dwell = np_dwell[cur] if cur != 0 else 0
                if not is_solomon and cur == 0: arrival_time = np_sched[cid]
                else: arrival_time = prev_pred_time_epoch + pre_to_cur + prev_dwell
                
                if is_solomon:
                    if arrival_time > np_latest[cid]: continue
                    start_time = max(arrival_time, np_earliest[cid])
                    pre_to_dc, cur_to_dc = np_time[cur, 0], np_time[cid, 0]
                    new_dur = cur_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + (start_time - arrival_time) + np_dwell[cid]
                else:
                    if arrival_time < np_earliest[cid] or arrival_time > np_latest[cid]: continue
                    if cur == 0: new_dur = np_time[0, cid] + np_dwell[cid] + np_time[cid, 0]
                    else:
                        pre_to_dc, cur_to_dc = np_time[cur, 0], np_time[cid, 0]
                        new_dur = cur_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + np_dwell[cid]
                
                if new_dur > time_limit: continue
                
                if np_dist[cur, cid] < best_dist:
                    best_dist = np_dist[cur, cid]
                    best_idx = cid
                    
            if best_idx == -1: break
            
            # Update state
            pre_to_cur = np_time[cur, best_idx]
            prev_dwell = np_dwell[cur] if cur != 0 else 0
            if not is_solomon and cur == 0: arrival_time = np_sched[best_idx]
            else: arrival_time = prev_pred_time_epoch + pre_to_cur + prev_dwell
            
            if is_solomon:
                start_time = max(arrival_time, np_earliest[best_idx])
                pre_to_dc, cur_to_dc = np_time[cur, 0], np_time[best_idx, 0]
                cur_duration = cur_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + (start_time - arrival_time) + np_dwell[best_idx]
                prev_pred_time_epoch = start_time
            else:
                start_time = arrival_time
                if cur == 0: cur_duration = np_time[0, best_idx] + np_dwell[best_idx] + np_time[best_idx, 0]
                else:
                    pre_to_dc, cur_to_dc = np_time[cur, 0], np_time[best_idx, 0]
                    cur_duration = cur_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + np_dwell[best_idx]
                prev_pred_time_epoch = start_time
                
            cur_vol += np_volume[best_idx]
            routes_flat[ptr] = best_idx
            ptr += 1
            routes_len[v_idx] += 1
            unvisited[best_idx] = False
            unvisited_count -= 1
            cur = best_idx
            route_found = True
            
        v_idx += 1
        if not route_found and unvisited_count > 0:
            # Force visit one if no feasible
            cid = -1
            for i in range(1, n_nodes):
                if unvisited[i]:
                    cid = i; break
            if cid != -1:
                routes_flat[ptr] = cid
                ptr += 1
                routes_len[v_idx-1] += 1
                unvisited[cid] = False
                unvisited_count -= 1
                
    return routes_flat, routes_len, v_idx

@njit(cache=True)
def _njit_solution_cost(routes_flat, routes_len, v_count, dc_idx, np_dist, np_time, np_volume, np_dwell, np_earliest, np_latest, np_sched, capacity, time_limit, is_solomon, dc_departure_time):
    total_dist = 0.0
    feasible = True
    ptr = 0
    total_visited = 0
    for v in range(v_count):
        l = routes_len[v]
        if l == 0: continue
        total_visited += l
        route = routes_flat[ptr:ptr+l]
        ptr += l
        f, d, _ = _njit_eval_route(route, np.zeros(0), dc_idx, np_dist, np_time, np_volume, np_dwell, np_earliest, np_latest, np_sched, capacity, time_limit, is_solomon, dc_departure_time)
        if not f:
            feasible = False
        total_dist += d
    
    # Check if all nodes visited (assuming n_nodes-1 stores)
    # This check is usually handled by the caller or specialized logic in MACS
    return feasible, total_dist

def _vei_worker(self, vei_max_vehicles, result_queue, stop_event, print_lock, turn_control):
    ph_vei = self.ph_vei # Use shared array
    in_vei = self.np_in_vei
    iter_num = 0
    
    while not stop_event.is_set():
        best_visited = self.best_visited_vei
        best_routes = None
        best_cost = (float('inf'), float('inf')) if self.is_solomon else float('inf')
        
        for _ in range(self.num_ants):
            if stop_event.is_set(): break
            r_vals_q = np.random.rand(self.store_count * 2)
            r_vals_r = np.random.rand(self.store_count * 2)
            
            r_flat, r_len, v_cnt = _njit_build_ant(
                self.store_count + 1, ph_vei, in_vei, self.np_dist, self.np_time, 
                self.np_volume, self.np_dwell, self.np_earliest, self.np_latest, self.np_sched, 
                self.np_group, self.np_region, self.dc_departure_time, 
                self.support_capacity, self.time_limit_per_route, 
                self.alpha, self.beta, self.rho, self.q0, self.tau0, 
                self.is_solomon, vei_max_vehicles, r_vals_q, r_vals_r
            )
            
            routes = []
            visited = set()
            p = 0
            for i in range(v_cnt):
                if r_len[i] > 0:
                    rt = list(r_flat[p:p+r_len[i]])
                    routes.append(rt); visited.update(rt)
                p += r_len[i]
            
            for i in range(1, self.store_count + 1):
                if i not in visited: in_vei[i] += 1.0
            
            nv = len(visited)
            if nv > best_visited:
                best_visited = nv
                self.best_visited_vei = nv
                in_vei.fill(0.0)
                cost = self._calc_cost(routes)
                best_routes, best_cost = routes, cost
                if nv == self.store_count:
                    result_queue.put(('VEI_FEASIBLE', routes, cost))
            elif nv == self.store_count:
                cost = self._calc_cost(routes)
                if self.is_solomon:
                    if cost[0] < best_cost[0] or (cost[0] == best_cost[0] and cost[1] < best_cost[1]):
                        best_routes, best_cost = routes, cost
                else:
                    if cost < best_cost:
                        best_routes, best_cost = routes, cost
        
        # Apply VND only to the best candidate of this cycle
        if best_routes and nv == self.store_count:
            best_routes = self._apply_vnd(best_routes)
            best_cost = self._calc_cost(best_routes)
            result_queue.put(('VEI_IMPROVED', best_routes, best_cost))

        if best_routes:
            _flat = np.array([n for r in best_routes for n in r], dtype=np.int64)
            _len = np.array([len(r) for r in best_routes], dtype=np.int64)
            c_val = best_cost[0] * self.vehicle_cost + best_cost[1] if self.is_solomon else best_cost
            _njit_global_update(ph_vei, _flat, _len, len(best_routes), c_val, self.rho)
            
        # Turn control print
        while turn_control[0] != 0 and not stop_event.is_set():
            time.sleep(0.01)
        if stop_event.is_set(): break
        
        with print_lock:
            if best_routes is not None and best_visited == self.store_count:
                c_disp = best_cost[1] if self.is_solomon else best_cost
                print(f"    [ACS-VEI]  iter {iter_num}, vehicle={len(best_routes)}, cost={c_disp:.2f} (FEASIBLE)")
            else:
                print(f"    [ACS-VEI]  iter {iter_num}, nodes={best_visited}/{self.store_count}")
            turn_control[0] = 1
        iter_num += 1

def _dist_worker(self, vei_max_vehicles, result_queue, stop_event, print_lock, turn_control):
    ph_dist = self.ph_dist
    iter_num = 0
    
    while not stop_event.is_set():
        best_routes = None
        best_cost = (float('inf'), float('inf')) if self.is_solomon else float('inf')
        max_v = 0

        for _ in range(self.num_ants):
            if stop_event.is_set(): break
            r_vals_q = np.random.rand(self.store_count * 2)
            r_vals_r = np.random.rand(self.store_count * 2)
            
            r_flat, r_len, v_cnt = _njit_build_ant(
                self.store_count + 1, ph_dist, np.zeros(self.store_count + 1, dtype=np.float64), self.np_dist, self.np_time, 
                self.np_volume, self.np_dwell, self.np_earliest, self.np_latest, self.np_sched, 
                self.np_group, self.np_region, self.dc_departure_time, 
                self.support_capacity, self.time_limit_per_route, 
                self.alpha, self.beta, self.rho, self.q0, self.tau0, 
                self.is_solomon, vei_max_vehicles, r_vals_q, r_vals_r
            )
            
            routes = []
            visited_cnt = 0
            p = 0
            for i in range(v_cnt):
                if r_len[i] > 0: 
                    routes.append(list(r_flat[p:p+r_len[i]]))
                    visited_cnt += r_len[i]
                p += r_len[i]
            
            if visited_cnt > max_v: max_v = visited_cnt
            
            if visited_cnt == self.store_count:
                cost = self._calc_cost(routes)
                if self.is_solomon:
                    if cost[0] < best_cost[0] or (cost[0] == best_cost[0] and cost[1] < best_cost[1]):
                        best_routes, best_cost = routes, cost
                else:
                    if cost < best_cost:
                        best_routes, best_cost = routes, cost
        
        # Apply VND only once per cycle
        if best_routes:
            best_routes = self._apply_vnd(best_routes)
            best_cost = self._calc_cost(best_routes)
            result_queue.put(('TIME_IMPROVED', best_routes, best_cost))
        
        if best_routes:
            _flat = np.array([n for r in best_routes for n in r], dtype=np.int64)
            _len = np.array([len(r) for r in best_routes], dtype=np.int64)
            c_val = best_cost[0] * self.vehicle_cost + best_cost[1] if self.is_solomon else best_cost
            _njit_global_update(ph_dist, _flat, _len, len(best_routes), c_val, self.rho)
            
        # Turn control print
        while turn_control[0] != 1 and not stop_event.is_set():
            time.sleep(0.01)
        if stop_event.is_set(): break
        
        with print_lock:
            if best_routes is not None:
                c_disp = best_cost[1] if self.is_solomon else best_cost
                print(f"    [ACS-DIST] iter {iter_num}, vehicle={len(best_routes)}, cost={c_disp:.2f}")
            else:
                print(f"    [ACS-DIST] iter {iter_num}, nodes={max_v}/{self.store_count}")
            turn_control[0] = 0
        iter_num += 1

@njit(cache=True)
def _njit_transition_value(current_idx, next_idx, in_next, cur_time, np_dist, np_time, np_earliest, np_latest, is_solomon, ph, alpha, beta):
    tau = ph[current_idx, next_idx]
    
    # MACS Transition Value (Eta) Logic
    travel_time = np_time[current_idx, next_idx]
    arrival = cur_time + travel_time
    delivery = max(arrival, np_earliest[next_idx]) if is_solomon else arrival
    
    delta = delivery - cur_time
    # d = delta * max(1.0, (due_date - cur_time))
    d = delta * max(1.0, float(np_latest[next_idx] - cur_time))
    
    # Incorporate IN vector (penalty/bonus for nodes not visited)
    d = max(1.0, d - in_next)
    
    eta = 1.0 / d
    return (tau ** alpha) * (eta ** beta)

@njit(cache=True)
def _njit_macs_choose(current_idx, cand_arr, in_vec, cur_time, np_dist, np_time, np_earliest, np_latest, is_solomon, ph, alpha, beta, q0, rand_q, rand_r):
    n_feas = len(cand_arr)
    if n_feas == 0:
        return -1
    if n_feas == 1:
        return cand_arr[0]

    # q0 mechanism: Exploitation vs Exploration
    if rand_q < q0:
        best_val = -1.0
        best_idx = -1
        for i in range(n_feas):
            next_idx = cand_arr[i]
            val = _njit_transition_value(current_idx, next_idx, in_vec[next_idx], cur_time, np_dist, np_time, np_earliest, np_latest, is_solomon, ph, alpha, beta)
            if val > best_val:
                best_val = val
                best_idx = next_idx
        return best_idx

    # Exploration (Roulette Wheel)
    probs = np.zeros(n_feas, dtype=np.float64)
    sum_prob = 0.0
    for i in range(n_feas):
        next_idx = cand_arr[i]
        val = _njit_transition_value(current_idx, next_idx, in_vec[next_idx], cur_time, np_dist, np_time, np_earliest, np_latest, is_solomon, ph, alpha, beta)
        probs[i] = val
        sum_prob += val

    if sum_prob == 0:
        return cand_arr[0]

    r = rand_r * sum_prob
    cum = 0.0
    for i in range(n_feas):
        cum += probs[i]
        if r <= cum:
            return cand_arr[i]
    return cand_arr[-1]

@njit(cache=True)
def _njit_get_feasible_stores(unvisited_indices, last_idx, route_vol, curr_duration,
                               prev_pred_time_epoch, dc_idx, support_capacity, time_limit,
                               dist_group, region, volume, time_matrix, dwell_time, 
                               earliest_time, latest_time, sched_time, is_solomon):
    feasible = []
    last_g = dist_group[last_idx]
    last_r = region[last_idx]
    
    for i in range(len(unvisited_indices)):
        store_idx = unvisited_indices[i]
        store_g = dist_group[store_idx]
        store_r = region[store_idx]
        
        # Region constraint
        if not is_solomon:
            if last_g == 2:
                if (last_r == 0 and store_r == 1) or \
                   (last_r == 1 and store_r == 0) or \
                   (last_r == 2 and store_r == 3) or \
                   (last_r == 3 and store_r == 2):
                    continue
                    
            if last_g == 2 and store_g not in (0, 1, 2):
                continue
            elif last_g == 1 and store_g not in (0, 1):
                continue
            elif last_g == 0 and store_g != 0:
                continue
            
        # Capacity
        if route_vol + volume[store_idx] > support_capacity:
            continue
            
        # Time constraints
        pre_to_cur = time_matrix[last_idx, store_idx]
        prev_dwell = dwell_time[last_idx] if last_idx != dc_idx else 0
        
        if not is_solomon and last_idx == dc_idx:
            arrival_time = sched_time[store_idx]
        else:
            arrival_time = prev_pred_time_epoch + pre_to_cur + prev_dwell
        
        if is_solomon:
            if arrival_time > latest_time[store_idx]:
                continue
                
            start_time = max(arrival_time, earliest_time[store_idx])
            pre_to_dc = time_matrix[last_idx, dc_idx]
            cur_to_dc = time_matrix[store_idx, dc_idx]
            new_duration = curr_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + (start_time - arrival_time) + dwell_time[store_idx]
        else:
            if arrival_time < earliest_time[store_idx]:
                continue
            if arrival_time > latest_time[store_idx]:
                continue

            if last_idx == dc_idx:
                new_duration = time_matrix[dc_idx, store_idx] + dwell_time[store_idx] + time_matrix[store_idx, dc_idx]
            else:
                pre_to_dc = time_matrix[last_idx, dc_idx]
                cur_to_dc = time_matrix[store_idx, dc_idx]
                new_duration = curr_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + dwell_time[store_idx]

        if new_duration > time_limit:
            continue
            
        feasible.append(store_idx)
        
    # Convert feasible list to array manually since Numba list appending can be slow/tricky
    res = np.zeros(len(feasible), dtype=np.int64)
    for i in range(len(feasible)):
        res[i] = feasible[i]
    return res

@njit(cache=True)
def _njit_build_ant(n_nodes, ph, in_vec, np_dist, np_time, np_volume, np_dwell, np_earliest, np_latest, np_sched, 
                    np_group, np_region, dc_departure_time, capacity, time_limit, 
                    alpha, beta, rho, q0, tau0, is_solomon, max_vehicles, rand_vals_q, rand_vals_r):
    
    unvisited = np.ones(n_nodes, dtype=np.bool_)
    unvisited[0] = False
    unvisited_count = n_nodes - 1
    
    routes_flat = np.zeros(n_nodes * 2, dtype=np.int64)
    routes_len = np.zeros(n_nodes, dtype=np.int64)
    
    ptr = 0
    v_idx = 0
    r_val_idx = 0
    
    while unvisited_count > 0 and v_idx < max_vehicles:
        cur = 0
        cur_vol = 0.0
        cur_duration = 0.0
        prev_pred_time_epoch = dc_departure_time if not is_solomon else 0
        
        while unvisited_count > 0:
            # Extract unvisited indices
            unv_arr = np.zeros(unvisited_count, dtype=np.int64)
            u_idx = 0
            for i in range(1, n_nodes):
                if unvisited[i]:
                    unv_arr[u_idx] = i
                    u_idx += 1
            
            feasible = _njit_get_feasible_stores(unv_arr, cur, cur_vol, cur_duration, prev_pred_time_epoch, 
                                                 0, capacity, time_limit, np_group, np_region, np_volume, 
                                                 np_time, np_dwell, np_earliest, np_latest, np_sched, is_solomon)
            
            if len(feasible) == 0:
                if cur == 0:
                    # Forced assignment (single store route)
                    chosen = unv_arr[0]
                else:
                    break
            else:
                rand_q = rand_vals_q[r_val_idx % len(rand_vals_q)]
                rand_r = rand_vals_r[r_val_idx % len(rand_vals_r)]
                r_val_idx += 1
                
                departure_time = prev_pred_time_epoch + (np_dwell[cur] if cur > 0 else 0)
                if not is_solomon and cur == 0:
                    # In non-Solomon mode, vehicle starts at sched_time of first store
                    # But for heuristic purpose, we can use dc_departure_time
                    departure_time = dc_departure_time

                chosen = _njit_macs_choose(cur, feasible, in_vec, departure_time, np_dist, np_time, np_earliest, np_latest, is_solomon, ph, alpha, beta, q0, rand_q, rand_r)
            
            # Local update
            ph[cur, chosen] = (1.0 - rho) * ph[cur, chosen] + rho * tau0
            ph[chosen, cur] = (1.0 - rho) * ph[chosen, cur] + rho * tau0
            
            # Update route state
            pre_to_cur = np_time[cur, chosen]
            prev_dwell = np_dwell[cur] if cur != 0 else 0
            
            if not is_solomon and cur == 0:
                arrival_time = np_sched[chosen]
            else:
                arrival_time = prev_pred_time_epoch + pre_to_cur + prev_dwell
            
            if is_solomon:
                start_time = max(arrival_time, np_earliest[chosen])
                pre_to_dc = np_time[cur, 0]
                cur_to_dc = np_time[chosen, 0]
                cur_duration = cur_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + (start_time - arrival_time) + np_dwell[chosen]
                prev_pred_time_epoch = start_time
            else:
                start_time = arrival_time
                if cur == 0:
                    cur_duration = np_time[0, chosen] + np_dwell[chosen] + np_time[chosen, 0]
                else:
                    pre_to_dc = np_time[cur, 0]
                    cur_to_dc = np_time[chosen, 0]
                    cur_duration = cur_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + np_dwell[chosen]
                prev_pred_time_epoch = start_time
                
            cur_vol += np_volume[chosen]
            
            routes_flat[ptr] = chosen
            ptr += 1
            routes_len[v_idx] += 1
            unvisited[chosen] = False
            unvisited_count -= 1
            cur = chosen
            
        v_idx += 1
        
    return routes_flat, routes_len, v_idx

@njit(cache=True)
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
# 2. Python Class
# ─────────────────────────────────────────────────────────────────────────────

class SupportLinePlanningMACS:
    def __init__(self, remaining_stores, distance_matrix, time_matrix,
                 num_ants=10, time_limit=60, support_capacity=7.2,
                 time_limit_per_route=5 * 60 * 60, vehicle_cost=2000, is_solomon=False,
                 alpha=1.0, beta=1.0, rho=0.1, q0=0.9, early_stop_patience=10,
                 verbose=True, vnd_strategy='best'):

        self.remaining_stores = remaining_stores
        self.orig_distance_matrix = distance_matrix
        self.orig_time_matrix = time_matrix
        self.verbose = verbose
        self.time_limit = time_limit
        self.is_solomon = is_solomon
        self.early_stop_patience = early_stop_patience
        self.vnd_strategy = vnd_strategy
        self.support_capacity = float(support_capacity)
        self.time_limit_per_route = float(time_limit_per_route)
        self.vehicle_cost = float(vehicle_cost)
        self.store_count = len(remaining_stores)
        
        self.num_ants = max(num_ants, len(remaining_stores))
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        
        self.dc_config = DC_CONFIG.copy()
        self.dc_config['max_capacity'] = self.support_capacity

        self._prepare_numpy_data()
        
        # MACS initial pheromone tau0
        gb_cost = self._greedy_cost()
        self.tau0 = 1.0 / (self.store_count * gb_cost) if self.store_count > 0 else 0.001
        
        self.ph_vei = np.full((self.store_count + 1, self.store_count + 1), self.tau0, dtype=np.float64)
        self.ph_dist = np.full((self.store_count + 1, self.store_count + 1), self.tau0, dtype=np.float64)
        for i in range(self.store_count + 1):
            self.ph_vei[i, i] = 0.0
            self.ph_dist[i, i] = 0.0
            
        self.log = []
        self.vnd = VND(distance_matrix, time_matrix, vehicle_cost=0, is_solomon=is_solomon, 
                       vnd_strategy=self.vnd_strategy, time_limit=time_limit_per_route)
        
        # Initial Solution
        if self.store_count > 0:
            nn_flat, nn_len, nn_cnt = _njit_nearest_neighbor(
                self.store_count + 1, self.np_dist, self.np_time, self.np_volume, self.np_dwell, 
                self.np_earliest, self.np_latest, self.np_sched, self.np_group, self.np_region, 
                self.dc_departure_time, self.support_capacity, self.time_limit_per_route, self.is_solomon
            )
            nn_routes = []
            p = 0
            for i in range(nn_cnt):
                if nn_len[i] > 0: nn_routes.append(list(nn_flat[p:p+nn_len[i]]))
                p += nn_len[i]
            self.gb_routes = nn_routes
            self.gb_cost = self._calc_cost(nn_routes)
            if self.verbose:
                c_disp = self.gb_cost[1] if self.is_solomon else self.gb_cost
                print(f"    [MACS] Init (NN): {len(self.gb_routes)} vehicles, cost={c_disp:.2f}")
        else:
            self.gb_routes = []
            self.gb_cost = (0, 0.0) if self.is_solomon else 0.0

    def _prepare_numpy_data(self):
        n = self.store_count + 1
        self.s2i = {self.dc_config.get('store_id', 'dc'): 0}
        self.i2s = {0: self.dc_config}
        
        store_ids = sorted(self.remaining_stores, key=lambda s: int(s.get('store_id', 0)) if str(s.get('store_id')).isdigit() else str(s.get('store_id')))
        
        for idx, s in enumerate(store_ids, start=1):
            self.s2i[s['store_id']] = idx
            self.i2s[idx] = s

        self.np_dist = np.zeros((n, n), dtype=np.float64)
        self.np_time = np.zeros((n, n), dtype=np.float64)
        self.np_volume = np.zeros(n, dtype=np.float64)
        self.np_dwell = np.zeros(n, dtype=np.int64)
        self.np_earliest = np.zeros(n, dtype=np.int64)
        self.np_latest = np.zeros(n, dtype=np.int64)
        self.np_sched = np.zeros(n, dtype=np.int64)
        self.np_group = np.full(n, -1, dtype=np.int64)
        self.np_region = np.full(n, -1, dtype=np.int64)
        self.np_in_vei = np.zeros(n, dtype=np.float64)
        self.best_visited_vei = 0
        
        region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        group_map = {'near': 0, 'mid': 1, 'far': 2}
        
        base_dt = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(n):
            s_i_id = self.i2s[i].get('store_id', 'dc')
            if i > 0:
                s_i = self.i2s[i]
                self.np_volume[i] = float(s_i.get('volume', 0.0))
                self.np_dwell[i] = int(s_i.get('dwell_time', 0))
                
                rt = datetime.fromisoformat(s_i['earliest_time'])
                dt = datetime.fromisoformat(s_i['latest_time'])
                st = datetime.fromisoformat(s_i.get('sched_time', s_i['earliest_time']))
                
                if self.is_solomon:
                    self.np_earliest[i] = int((rt - base_dt).total_seconds() / 60)
                    self.np_latest[i] = int((dt - base_dt).total_seconds() / 60)
                    self.np_sched[i] = self.np_earliest[i]
                else:
                    self.np_earliest[i] = int(rt.timestamp())
                    self.np_latest[i] = int(dt.timestamp())
                    self.np_sched[i] = int(st.timestamp())
                
                self.np_group[i] = group_map.get(s_i.get('dist_group', ''), -1)
                self.np_region[i] = region_map.get(s_i.get('region', ''), -1)
                
            for j in range(n):
                s_j_id = self.i2s[j].get('store_id', 'dc')
                self.np_dist[i, j] = float(self.orig_distance_matrix.get(s_i_id, {}).get(s_j_id, 0.0))
                self.np_time[i, j] = float(self.orig_time_matrix.get(s_i_id, {}).get(s_j_id, 0.0))

        if not self.is_solomon and self.store_count > 0:
            self.dc_departure_time = int(np.min(self.np_earliest[1:self.store_count + 1]))
        else:
            self.dc_departure_time = 0

    def _greedy_cost(self):
        unvisited = set(range(1, self.store_count + 1))
        cost = 0.0
        while unvisited:
            cur = 0
            while unvisited:
                best_dist = float('inf')
                best = None
                for cid in unvisited:
                    if self.np_dist[cur, cid] < best_dist:
                        best_dist = self.np_dist[cur, cid]
                        best = cid
                if best is None: break
                cost += best_dist
                unvisited.remove(best)
                cur = best
            cost += self.np_dist[cur, 0]
        return cost if cost > 0 else 1.0

    def _calc_cost(self, routes):
        total_dist = 0.0
        for route in routes:
            if not route: continue
            total_dist += self.np_dist[0, route[0]]
            for i in range(len(route)-1):
                total_dist += self.np_dist[route[i], route[i+1]]
            total_dist += self.np_dist[route[-1], 0]
            
        if self.is_solomon:
            return (len(routes), total_dist)
        return total_dist + (len(routes) * self.vehicle_cost)

    def _format_solution(self, routes):
        base_dt = datetime(2024, 1, 1, 0, 0, 0)
        res = {}
        for v_id, route in enumerate(routes):
            r_id = f"{101 + v_id}"
            
            total_dist = 0.0
            total_load = 0.0
            cur_duration = 0.0
            prev_pred_time_epoch = self.dc_departure_time if not self.is_solomon else 0
            prev = 0
            
            formatted_stores = []
            for i, idx in enumerate(route):
                s_info = self.i2s[idx].copy()
                
                travel_dist = self.np_dist[prev, idx]
                travel_time = self.np_time[prev, idx]
                prev_dwell = self.np_dwell[prev] if prev != 0 else 0
                
                if not self.is_solomon and prev == 0:
                    arrival = self.np_sched[idx]
                else:
                    arrival = prev_pred_time_epoch + travel_time + prev_dwell
                
                if self.is_solomon:
                    start = max(arrival, self.np_earliest[idx])
                    s_info['pred_time'] = (base_dt + timedelta(minutes=start)).isoformat(timespec='seconds')
                    cur_duration = cur_duration + (travel_time + self.np_time[idx, 0] - self.np_time[prev, 0]) + (start - arrival) + self.np_dwell[idx]
                else:
                    start = arrival
                    s_info['pred_time'] = datetime.fromtimestamp(start).isoformat(timespec='seconds')
                    if prev == 0:
                        cur_duration = self.np_time[0, idx] + self.np_dwell[idx] + self.np_time[idx, 0]
                    else:
                        cur_duration = cur_duration + (travel_time + self.np_time[idx, 0] - self.np_time[prev, 0]) + self.np_dwell[idx]
                
                s_info['route_id'] = r_id
                formatted_stores.append(s_info)
                
                prev_pred_time_epoch = start
                total_dist += travel_dist
                total_load += self.np_volume[idx]
                prev = idx
                
            total_dist += self.np_dist[prev, 0]
            
            dc = {
                "route_id": r_id,
                "route_code": r_id,
                "store_id": self.dc_config.get('store_id', 'dc'),
                "store_name": self.dc_config.get('store_name', 'Solomon Depot'),
                "total_volume": float(total_load),
                "load_rate": float(total_load / self.support_capacity) if self.support_capacity > 0 else 0.0,
                "max_capacity": float(self.support_capacity),
                "region": self.dc_config.get('region', 'unknown'),
                "distance": float(total_dist),
                "duration": float(cur_duration)
            }
            
            res[r_id] = {"dc": dc, "stores": formatted_stores}
        return res

    def _apply_vnd(self, routes):
        routes_info = self._format_solution(routes)
        opt_routes_info, _ = self.vnd.optimize(routes_info)
        
        new_routes = []
        for k, v in opt_routes_info.items():
            if not v['stores']: continue
            new_routes.append([self.s2i[s['store_id']] for s in v['stores']])
        return new_routes

    def run(self):
        if self.store_count == 0:
            return self.gb_cost, self._format_solution(self.gb_routes)
        
        t_start = time.time()
        early_stopper = EarlyStopper(patience=self.early_stop_patience)
        global_iter = 0
        
        while (time.time() - t_start) < self.time_limit:
            v = len(self.gb_routes)
            if self.verbose:
                c_disp = self.gb_cost[1] if self.is_solomon else self.gb_cost
                print(f"    [MACS] iteration {global_iter}: vehicle={v}, cost={c_disp:.2f}, elapsed={time.time()-t_start:.1f}s")
            
            # Initial log for each outer iteration
            self.log.append({
                'iteration': global_iter,
                'vehicle_count': v,
                'iter_best_cost': self.gb_cost[1] if self.is_solomon else self.gb_cost,
                'best_cost': self.gb_cost[1] if self.is_solomon else self.gb_cost,
                'elapsed_time': time.time() - t_start
            })
            
            res_q = queue.Queue()
            stop_event = threading.Event()
            print_lock = threading.Lock()
            turn_control = [0] # 0: VEI, 1: DIST
            
            t_vei = threading.Thread(target=_vei_worker, args=(self, max(1, v - 1), res_q, stop_event, print_lock, turn_control))
            t_dist = threading.Thread(target=_dist_worker, args=(self, max(1, v),res_q, stop_event, print_lock, turn_control))
            
            t_vei.start(); t_dist.start()
            
            # Monitoring loop
            while t_vei.is_alive() or t_dist.is_alive():
                if (time.time() - t_start) >= self.time_limit:
                    stop_event.set()
                    break
                try:
                    msg = res_q.get(timeout=0.1)
                    src, sol, cost = msg
                    
                    improved = False
                    if self.is_solomon:
                        if cost[0] < self.gb_cost[0] or (cost[0] == self.gb_cost[0] and cost[1] < self.gb_cost[1]):
                            old_nv = self.gb_cost[0]
                            self.gb_routes, self.gb_cost = sol, cost
                            improved = True
                            if cost[0] < old_nv:
                                if self.verbose: print(f"    [MACS] VEI reduced vehicles! {cost[0]}")
                                stop_event.set(); break
                    else:
                        if cost < self.gb_cost:
                            old_nv = len(self.gb_routes)
                            self.gb_routes, self.gb_cost = sol, cost
                            improved = True
                            if len(sol) < old_nv:
                                if self.verbose: print(f"    [MACS] VEI reduced vehicles! {len(sol)}")
                                stop_event.set(); break
                    
                    if improved:
                        self.log.append({
                            'iteration': global_iter,
                            'vehicle_count': self.gb_cost[0] if self.is_solomon else len(self.gb_routes),
                            'iter_best_cost': self.gb_cost[1] if self.is_solomon else self.gb_cost,
                            'best_cost': self.gb_cost[1] if self.is_solomon else self.gb_cost,
                            'elapsed_time': time.time() - t_start
                        })
                except queue.Empty:
                    continue
            
            stop_event.set()
            t_vei.join(); t_dist.join()
            global_iter += 1
            
            if early_stopper.check(self.gb_cost):
                if self.verbose: print(f"    [MACS] Early stop triggered.")
                break
                
        elapsed = time.time() - t_start
        if self.verbose:
            c_disp = self.gb_cost[1] if self.is_solomon else self.gb_cost
            nv = self.gb_cost[0] if self.is_solomon else len(self.gb_routes)
            print(f"    [MACS] Done in {elapsed:.1f}s → {nv} vehicles, cost={c_disp:.2f}")
            
        return self.gb_cost, self._format_solution(self.gb_routes)
