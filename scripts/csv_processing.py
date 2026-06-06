import os
import pandas as pd
import openpyxl
from openpyxl.styles import Alignment

# --- Original Functions (for summary_results.csv) ---

def read_and_process_csv(file_path):
    """
    Reads and processes a standard summary CSV file.
    - Reads the CSV.
    - Removes specific columns ('Loss_f', 'Loss_g', 'dist').
    - Drops empty or all-NA columns.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, index_col=0)
        # Remove specific columns
        columns_to_remove = ['Loss_f', 'Loss_g', 'dist']
        df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        # Drop empty or all-NA columns
        df.dropna(axis=1, how='all', inplace=True)
        return df
    except Exception as e:
        print(f"Error reading or processing {file_path}: {e}")
        return pd.DataFrame()

def save_to_excel_with_formatting(df, output_excel_file, sheet_name='Metrics Summary'):
    """
    Saves a DataFrame to an Excel file with specific formatting.
    - Centers cell content.
    - Adjusts column widths.
    """
    try:
        with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=True, sheet_name=sheet_name)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # Set cell format to center align for all data cells and headers
            # Iterate through all columns including the index
            for col_idx, column_cells in enumerate(worksheet.columns, 1):
                for row_idx, cell in enumerate(column_cells, 1):
                    cell.alignment = Alignment(horizontal='center', vertical='center')

            # Set fixed column widths
            # Column A (index) width, then others
            for col_idx, column_letter in enumerate([cell.column_letter for cell in worksheet[1]], start=1):
                max_length = 0
                for cell in worksheet[column_letter]:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = 23 if col_idx == 1 else (max_length + 2 if max_length + 2 < 14 else 14)
                if max_length + 2 > 14 and col_idx > 1 : # For data columns wider than 14
                    adjusted_width = max_length + 2
                worksheet.column_dimensions[column_letter].width = adjusted_width
        print(f"Successfully saved and formatted: {output_excel_file}")
    except Exception as e:
        print(f"Error saving or formatting {output_excel_file}: {e}")


def combine_csv_files_with_titles(root_dir):
    """
    Processes 'summary_results.csv' from subfolders and saves each as a formatted Excel.
    """
    print("\n--- Processing original summary_results.csv files ---")
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            metrics_file = os.path.join(folder_path, 'eval', 'summary_results.csv')
            metrics_df = read_and_process_csv(metrics_file)

            if not metrics_df.empty:
                output_excel_file = os.path.join(folder_path, f'{folder}_combined_metrics_summary_with_titles.xlsx')
                save_to_excel_with_formatting(metrics_df, output_excel_file, sheet_name='Metrics Summary')
                # print(f"Combined Excel for {folder} (original) saved to {output_excel_file}") # Covered by save_to_excel
            else:
                if os.path.exists(metrics_file): # File exists but processing resulted in empty df
                     print(f"Processed file is empty for {folder}: {metrics_file}")
                # else: # File not found warning is handled by read_and_process_csv
                #    pass


def combine_all_excel_files_to_one_sheet(root_dir):
    """
    Combines all individual '*_combined_metrics_summary_with_titles.xlsx' files
    from subfolders into a single Excel sheet in the root directory.
    """
    print("\n--- Combining original processed Excel files ---")
    combined_output_file = os.path.join(root_dir, 'combined_all_folders.xlsx')
    combined_df = pd.DataFrame()

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            input_excel_file = os.path.join(folder_path, f'{folder}_combined_metrics_summary_with_titles.xlsx')
            if os.path.exists(input_excel_file):
                try:
                    df = pd.read_excel(input_excel_file, sheet_name='Metrics Summary', index_col=0)
                    df.insert(0, 'Folder', folder) # Insert folder name as the first column
                    combined_df = pd.concat([combined_df, df], ignore_index=False) # Keep original index
                except Exception as e:
                    print(f"Error reading {input_excel_file}: {e}")
            # else: # No need to print if intermediate file doesn't exist, previous step would have indicated issue
            #    print(f"Intermediate Excel file not found for {folder}: {input_excel_file}")


    if not combined_df.empty:
        save_to_excel_with_formatting(combined_df, combined_output_file, sheet_name='Combined Metrics Summary')
        # print(f"All original individual Excel files combined and saved to {combined_output_file}") # Covered by save_to_excel
    else:
        print("No data to combine for original summary files.")

# --- New Functions (for DEG_summary_results.csv) ---

def read_and_process_deg_csv(file_path):
    """
    Reads and processes a DEG summary CSV file.
    - Reads the CSV.
    - Drops empty or all-NA columns. (Does NOT remove 'Loss_f', 'Loss_g', 'dist')
    """
    if not os.path.exists(file_path):
        print(f"Warning: DEGs File not found - {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, index_col=0)
        # Drop empty or all-NA columns
        df.dropna(axis=1, how='all', inplace=True)
        return df
    except Exception as e:
        print(f"Error reading or processing DEGs file {file_path}: {e}")
        return pd.DataFrame()

# We can reuse save_to_excel_with_formatting or create a specific one if formatting needs to differ.
# For now, let's reuse it and pass a different sheet_name.
# If specific formatting for DEGs is needed later, a new function can be easily created.
# def save_deg_to_excel_with_formatting(df, output_excel_file, sheet_name='DEGs Metrics Summary'):
#    save_to_excel_with_formatting(df, output_excel_file, sheet_name) # Reusing the original

def combine_deg_csv_files_with_titles(root_dir):
    """
    Processes 'DEGs_summary_results.csv' from subfolders and saves each as a formatted Excel.
    """
    print("\n--- Processing DEGs_summary_results.csv files ---")
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            deg_metrics_file = os.path.join(folder_path, 'eval', 'DEGs_summary_results.csv')
            deg_metrics_df = read_and_process_deg_csv(deg_metrics_file)

            if not deg_metrics_df.empty:
                # New intermediate file name for DEGs data
                output_deg_excel_file = os.path.join(folder_path, f'{folder}_combined_deg_metrics_summary.xlsx')
                save_to_excel_with_formatting(deg_metrics_df, output_deg_excel_file, sheet_name='DEGs Metrics Summary')
                # print(f"Combined DEGs Excel for {folder} saved to {output_deg_excel_file}") # Covered by save_to_excel
            else:
                if os.path.exists(deg_metrics_file):
                    print(f"Processed DEGs file is empty for {folder}: {deg_metrics_file}")
                # else: # File not found warning is handled by read_and_process_deg_csv
                #    pass


def combine_all_deg_excel_files_to_one_sheet(root_dir):
    """
    Combines all individual '*_combined_deg_metrics_summary.xlsx' files
    from subfolders into a single Excel sheet in the root directory, named 'DEGs_combined_all_folders.xlsx'.
    """
    print("\n--- Combining DEGs processed Excel files ---")
    combined_deg_output_file = os.path.join(root_dir, 'DEGs_combined_all_folders.xlsx')
    combined_deg_df = pd.DataFrame()

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Input DEGs Excel file name
            input_deg_excel_file = os.path.join(folder_path, f'{folder}_combined_deg_metrics_summary.xlsx')
            if os.path.exists(input_deg_excel_file):
                try:
                    df = pd.read_excel(input_deg_excel_file, sheet_name='DEGs Metrics Summary', index_col=0)
                    df.insert(0, 'Folder', folder) # Insert folder name as the first column
                    combined_deg_df = pd.concat([combined_deg_df, df], ignore_index=False) # Keep original index
                except Exception as e:
                    print(f"Error reading DEGs Excel {input_deg_excel_file}: {e}")
            # else:
            #    print(f"Intermediate DEGs Excel file not found for {folder}: {input_deg_excel_file}")


    if not combined_deg_df.empty:
        save_to_excel_with_formatting(combined_deg_df, combined_deg_output_file, sheet_name='Combined DEGs Summary')
        # print(f"All DEGs individual Excel files combined and saved to {combined_deg_output_file}") # Covered by save_to_excel
    else:
        print("No data to combine for DEGs summary files.")


if __name__ == "__main__":
    # 设置根目录 - 请根据您的实际路径进行修改
    # Set the root directory - please modify it according to your actual path
    root_directory = '/home/jamin/condot/results/nine_drugs/977genes/drug_pert/CCOT'
    # root_directory = '.' # 或者设置为当前目录 for testing: or set to current directory for testing

    # 运行原有功能
    # Run original functionality
    combine_csv_files_with_titles(root_directory)
    combine_all_excel_files_to_one_sheet(root_directory)

    # 运行新添加的DEGs处理功能
    # Run newly added DEGs processing functionality
    combine_deg_csv_files_with_titles(root_directory)
    combine_all_deg_excel_files_to_one_sheet(root_directory)

    print("\nAll processing finished.")
