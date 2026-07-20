# From the Author
"""
This code is a data assembly and label integration tool. It helps you automatically 
merge, clean, and label raw sensor data scattered across dozens of folders and 
hundreds of CSV files, ultimately synthesizing it into a neatly structured Pandas DataFrame.
"""
import pandas as pd
import os


def get_ds_infos_as_dict():
    """
    Reads the subjects CSV file and converts it into a Python dictionary with 'code' as the Key.
    This allows us to look up each subject's real profile data using IDs 1–24 like a dictionary.
    """
    if not os.path.exists("data_subjects_info.csv"):
        raise FileNotFoundError("Error: 'data_subjects_info.csv' not found in the current directory. Please check the file location!")

    df_info = pd.read_csv("data_subjects_info.csv")

    # Force conversion of the 'code' column to integer
    df_info['code'] = df_info['code'].astype(int)

    # Print raw information for quick visual verification
    print("\n[DEBUG] -- Checking the content of your data_subjects_info.csv file:")
    print(df_info.head(3))  # Print first 3 subjects
    print(df_info.tail(3))  # Print last 3 subjects

    # Convert to dictionary structure
    info_dict = df_info.set_index('code').to_dict(orient='index')

    return info_dict


def set_data_types(data_types):
    """
    Configures the sensor data column names to extract (all 12 physical features).
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t + ".x", t + ".y", t + ".z"])
        else:
            dt_list.append([t + ".roll", t + ".pitch", t + ".yaw"])
    return dt_list


def create_perfect_dataset(dt_list, act_labels, trial_codes_list):
    # Flatten all physical feature columns (12 columns in total)
    flat_sensor_cols = [col for sublist in dt_list for col in sublist]

    # 1. Get the lookup dictionary for subject info
    subject_dict = get_ds_infos_as_dict()

    # Store all DataFrame chunks
    all_dfs = []

    print("\n[INFO] -- Starting to read and seamlessly concatenate time-series data...")

    # 2. Iterate through subjects 1 to 24
    for sub_id in range(1, 25):
        # Fetch current subject's profile data in real time via dictionary lookup
        current_sub_info = subject_dict.get(sub_id)

        # Skip if subject is not found in the dictionary
        if current_sub_info is None:
            continue

        weight = current_sub_info['weight']
        height = current_sub_info['height']
        age = current_sub_info['age']
        gender = current_sub_info['gender']

        # Debug log: clear feedback on who is currently being read
        print(f"[DEBUG] -- Reading time-series files for Subject ID: {sub_id} (Weight: {weight}kg, Height: {height}cm)")

        for act_id, act in enumerate(act_labels):
            # Fix: trial_codes_list is already sorted according to act_id
            for trial in trial_codes_list[act_id]:
                fname = f'A_DeviceMotion_data/{act}_{trial}/sub_{sub_id}.csv'

                if not os.path.exists(fname):
                    continue

                # Read current sensor data
                raw_data = pd.read_csv(fname)
                sensor_data = raw_data[flat_sensor_cols].copy()

                # 3. Explicitly attach target metadata to the current DataFrame block
                sensor_data["act"] = int(act_id)
                sensor_data["id"] = int(sub_id)
                sensor_data["weight"] = float(weight)
                sensor_data["height"] = float(height)
                sensor_data["age"] = float(age)
                sensor_data["gender"] = int(gender)
                sensor_data["trial"] = int(trial)

                all_dfs.append(sensor_data)

    if not all_dfs:
        print("[ERROR] -- Failed to read any sensor data. Please check the 'A_DeviceMotion_data' folder path!")
        return pd.DataFrame()

    # 4. Vertically concatenate all data
    print("\n[INFO] -- Vertically merging sensor time-series across all subjects...")
    final_dataset = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns (12 physical features + 5 label metadata attributes = 17 features total)
    ordered_cols = flat_sensor_cols + ["act", "id", "weight", "height", "age", "gender", "trial"]
    final_dataset = final_dataset[ordered_cols]

    return final_dataset


# ==================== Configuration & Run ====================

ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]: [1, 2, 11],
    ACT_LABELS[1]: [3, 4, 12],
    ACT_LABELS[2]: [7, 8, 15],
    ACT_LABELS[3]: [9, 16],
    ACT_LABELS[4]: [6, 14],
    ACT_LABELS[5]: [5, 13]
}

# Select all 4 main sensor categories (extracting full 12 features)
sdt = ["attitude", "gravity", "rotationRate", "userAcceleration"]
dt_list = set_data_types(sdt)

# Core Fix 2: Convert dictionary to an activity-aligned list to prevent KeyError
trial_codes_list = [TRIAL_CODES[act] for act in ACT_LABELS]

# Generate merged dataset
dataset = create_perfect_dataset(dt_list, ACT_LABELS, trial_codes_list)

if not dataset.empty:
    print("\n[INFO] -- Correctly merged dataset shape: " + str(dataset.shape))

    # Save output
    output_filename = "combined_devices_data.csv"
    print(f"[INFO] -- Saving dataset to: {output_filename} ...")
    dataset.to_csv(output_filename, index=False)
    print("[INFO] -- 🥳 Merged and saved successfully!")

    # 【Data Validation】
    print("\n--- 【Data Verification】 First 2 rows (Should show Subject ID 1 background, e.g., Weight 102kg) ---")
    print(dataset.head(2))

    print("\n--- 【Data Verification】 Transition rows where ID changes to 2 (Weight should change to 72kg) ---")
    print(dataset[dataset["id"] == 2].head(2))

    print("\n--- 【Data Verification】 Last 2 rows (Should show Subject ID 24 background, e.g., Weight 74kg) ---")
    print(dataset.tail(2))
else:
    print("[FAILED] -- Dataset creation failed. Please verify project paths and files.")