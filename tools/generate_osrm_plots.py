import json
import pandas as pd
import requests
import folium
import os
import sys

# 將 src 加入 Python 搜尋路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from config.config import OSRM_HOST

def get_osrm_route(coords):
    coord_str = ';'.join(f'{lon},{lat}' for lon, lat in coords)
    url = f'{OSRM_HOST}/route/v1/driving/{coord_str}?overview=full&geometries=geojson'
    try:
        res = requests.get(url).json()
        if 'routes' in res and len(res['routes']) > 0:
            return res['routes'][0]['geometry']['coordinates']
    except Exception as e:
        print('OSRM fetch error:', e)
    return coords

def jitter(coords, amount=0.0005):
    seen = {}
    new_coords = []
    for (lon, lat) in coords:
        key = (round(lon, 4), round(lat, 4))
        if key in seen:
            seen[key] += 1
            lon += amount * seen[key] * (1 if seen[key]%2==0 else -1)
            lat += amount * seen[key] * (1 if seen[key]%2==0 else -1)
        else:
            seen[key] = 0
        new_coords.append((lon, lat))
    return new_coords

def create_html(osrm_path, coords_list, store_names, store_info, title, out_path, depot_lon, depot_lat):
    center_lat = sum(c[1] for c in coords_list) / len(coords_list) if coords_list else depot_lat
    center_lon = sum(c[0] for c in coords_list) / len(coords_list) if coords_list else depot_lon
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
    
    if osrm_path:
        folium.PolyLine(locations=[[lat, lon] for lon, lat in osrm_path], color='blue', weight=4, opacity=0.8, tooltip=title).add_to(m)
        
    for i, (lon, lat) in enumerate(coords_list):
        s_name = store_names[i]
        rcode = store_info[s_name]['route_code']
        html = f'<b>Seq:</b> {i+1}<br><b>Route:</b> {rcode}<br><b>Name:</b> {s_name}'
        folium.CircleMarker(
            location=[lat, lon], radius=6, color='black', weight=1, fill=True, fill_color='red', fill_opacity=1,
            popup=folium.Popup(html, max_width=200)
        ).add_to(m)
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 10pt; font-weight: bold; color: black; background-color: rgba(255,255,255,0.7); border-radius: 3px; padding: 2px; white-space: nowrap;">{i+1} ({rcode})</div>',
                icon_size=(100, 20),
                icon_anchor=(-10, 10)
            )
        ).add_to(m)
        
    # Depot Marker
    folium.Marker(location=[depot_lat, depot_lon], popup='Depot', icon=folium.Icon(color='black', icon='star')).add_to(m)
    folium.map.Marker(
        [depot_lat, depot_lon],
        icon=folium.DivIcon(
            html=f'<div style="font-size: 11pt; font-weight: bold; color: black; background-color: rgba(255,255,255,0.9); border: 2px solid black; border-radius: 4px; padding: 3px; white-space: nowrap;">Depot</div>',
            icon_size=(60, 25),
            icon_anchor=(-15, 15)
        )
    ).add_to(m)
    m.save(out_path)

def plot_route_comparison(route_name, opt_path, man_path, output_dir, dataset_name="Dataset"):
    print(f"=== 正在處理 {dataset_name} 的 {route_name} 車次 ===")
    
    # 建立輸出目錄
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load Optimized Data
    with open(opt_path, 'r', encoding='utf-8') as f:
        opt_data = json.load(f)

    store_info = {}
    for r_id, info in opt_data.items():
        for s in info.get('stores', []):
            store_info[s['store_name']] = {'lon': s['longitude'], 'lat': s['latitude'], 'route_code': s['route_code']}

    opt_target_stores = []
    if route_name in opt_data:
        opt_target_stores = [s['store_name'] for s in opt_data[route_name].get('stores', [])]

    # Load Manual Data
    df_man = pd.read_excel(man_path, header=2)

    man_target_stores = []
    current_route = None
    for idx, row in df_man.iterrows():
        col0 = str(row.iloc[0])
        col1 = row.iloc[1]
        if col0.endswith('DC') or col0 == 'nan' or col0 == '車次': continue
        if pd.isna(col1):
            current_route = col0
        else:
            if current_route == route_name:
                man_target_stores.append(row.iloc[2])

    depot_lon, depot_lat = 121.370556, 25.0775

    man_coords = [(store_info[s]['lon'], store_info[s]['lat']) for s in man_target_stores if s in store_info]
    opt_coords = [(store_info[s]['lon'], store_info[s]['lat']) for s in opt_target_stores if s in store_info]

    man_coords = jitter(man_coords)
    opt_coords = jitter(opt_coords)

    man_full_coords = [(depot_lon, depot_lat)] + man_coords + [(depot_lon, depot_lat)]
    opt_full_coords = [(depot_lon, depot_lat)] + opt_coords + [(depot_lon, depot_lat)]

    print("Fetching OSRM routes...")
    man_osrm_path = get_osrm_route(man_full_coords) if man_coords else []
    opt_osrm_path = get_osrm_route(opt_full_coords) if opt_coords else []

    # ---- HTML Generation ----
    man_html_name = f"{dataset_name}_{route_name}_route_manual_osrm.html"
    opt_html_name = f"{dataset_name}_{route_name}_route_optimized_osrm.html"
    
    create_html(man_osrm_path, man_coords, man_target_stores, store_info, f'{dataset_name} {route_name} Manual', os.path.join(output_dir, man_html_name), depot_lon, depot_lat)
    create_html(opt_osrm_path, opt_coords, opt_target_stores, store_info, f'{dataset_name} {route_name} Proposed', os.path.join(output_dir, opt_html_name), depot_lon, depot_lat)

    print(f"完成！地圖已儲存至: {output_dir}")


if __name__ == "__main__":
    # ==========================================
    # 在這裡填寫你的資料路徑與想要繪製的路線
    # ==========================================
    
    # 1. 幫你的實驗結果取個名字 (會顯示在地圖檔名與標題上)
    DATASET_NAME = "D20260102"
    
    # 2. 想要分析的路線名稱 (例如 '2M', '3A', '5H')
    TARGET_ROUTE = "2M"
    
    # 3. 最佳化結果的路徑 (JSON檔)
    OPTIMIZED_JSON_PATH = os.path.join("output", "20260102", "2026-05-28_21-48-53", "optimized_routes_info.json")
    
    # 4. 手動編排結果的路徑 (Excel檔)
    MANUAL_EXCEL_PATH = os.path.join("output", "20260102", "manual_routes", "manual_routes_info.xlsx")
    
    # 5. 輸出的資料夾位置 (統一放在 output/route_plots 裡面)
    OUTPUT_DIRECTORY = os.path.join("output", "route_plots")
    
    # ==========================================
    
    plot_route_comparison(
        route_name=TARGET_ROUTE,
        opt_path=OPTIMIZED_JSON_PATH,
        man_path=MANUAL_EXCEL_PATH,
        output_dir=OUTPUT_DIRECTORY,
        dataset_name=DATASET_NAME
    )
