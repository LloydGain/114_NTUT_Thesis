import pandas as pd
import numpy as np
import os

def process_excel(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    df = pd.read_excel(input_path, header=None)

    manual_data = []
    program_data = []

    # Row chunks: 0, 8, 16, 24, 32, 40...
    for start_row in range(0, len(df), 8):
        if start_row + 6 >= len(df):
            break
        
        # Check if this chunk has labels in col 0
        label = str(df.iloc[start_row + 2, 0]).strip()
        if label != 'vehicle num':
            continue
            
        for col_idx in range(1, df.shape[1], 2):
            date = df.iloc[start_row, col_idx]
            if pd.isna(date):
                # Try previous column if it's the even one of a pair
                date = df.iloc[start_row, col_idx - 1] if col_idx > 1 else np.nan
            
            if pd.isna(date):
                continue
                
            # Date column is col_idx, so col_idx is Manual, col_idx+1 is Programmatic
            j = col_idx
            
            # Manual (j)
            if j < df.shape[1] and not pd.isna(df.iloc[start_row + 2, j]):
                manual_data.append({
                    'date': date,
                    'vehicle num': df.iloc[start_row + 2, j],
                    'total_dist(km)': df.iloc[start_row + 3, j],
                    'total_time(hr)': df.iloc[start_row + 4, j],
                    'avg_load_rate': df.iloc[start_row + 5, j],
                    'on_time_rate': df.iloc[start_row + 6, j]
                })
                
            # Programmatic (j+1)
            if j + 1 < df.shape[1] and not pd.isna(df.iloc[start_row + 2, j + 1]):
                program_data.append({
                    'date': date,
                    'vehicle num': df.iloc[start_row + 2, j + 1],
                    'total_dist(km)': df.iloc[start_row + 3, j + 1],
                    'total_time(hr)': df.iloc[start_row + 4, j + 1],
                    'avg_load_rate': df.iloc[start_row + 5, j + 1],
                    'on_time_rate': df.iloc[start_row + 6, j + 1]
                })

    df_manual = pd.DataFrame(manual_data)
    df_program = pd.DataFrame(program_data)

    # Helper function to format dataframe
    def format_df(df):
        if df.empty:
            return df
        
        numeric_cols = ['vehicle num', 'total_dist(km)', 'total_time(hr)', 'avg_load_rate', 'on_time_rate']
        
        # Ensure numeric types and round to 2 decimal places
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['date'] = df['date'].dt.strftime('%Y%m%d')

        # Calculate averages for numeric columns
        avg_values = df[numeric_cols].mean().round(2)
        
        # Create average row
        avg_row_data = {'date': 'Average'}
        for col in numeric_cols:
            avg_row_data[col] = avg_values[col]
            
        avg_row = pd.DataFrame([avg_row_data])
        
        # Append average row
        df = pd.concat([df, avg_row], ignore_index=True)
        return df

    df_manual = format_df(df_manual)
    df_program = format_df(df_program)

    with pd.ExcelWriter(output_path) as writer:
        if not df_manual.empty:
            df_manual.to_excel(writer, sheet_name='手動編排', index=False)
        if not df_program.empty:
            df_program.to_excel(writer, sheet_name='程式編排', index=False)

    print(f"Success! Organized Excel saved to: {output_path}")

if __name__ == "__main__":
