import json

def fix_missing_physical_data(input_file, output_file):
    # 1. Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    players = data.get('players', [])
    
    # 2. Gather all valid (greater than 0) heights and weights
    valid_heights = [p.get('height', 0) for p in players if p.get('height', 0) > 0]
    valid_weights = [p.get('weight', 0) for p in players if p.get('weight', 0) > 0]
    
    # 3. Calculate the averages (defaulting to 180cm / 75kg if no valid data exists)
    avg_height = int(sum(valid_heights) / len(valid_heights)) if valid_heights else 180
    avg_weight = int(sum(valid_weights) / len(valid_weights)) if valid_weights else 75
    
    print(f"📊 Calculated Average Height: {avg_height} cm")
    print(f"📊 Calculated Average Weight: {avg_weight} kg")
    
    # 4. Iterate through players and fix missing values
    fixed_height_count = 0
    fixed_weight_count = 0
    fixed_foot_count = 0
    
    for p in players:
        # Fix Height
        if p.get('height', 0) == 0:
            p['height'] = avg_height
            fixed_height_count += 1
            
        # Fix Weight
        if p.get('weight', 0) == 0:
            p['weight'] = avg_weight
            fixed_weight_count += 1
            
        # Fix Preferred Foot (JSON 'null' becomes Python 'None')
        if p.get('foot') is None:
            p['foot'] = 'right'
            fixed_foot_count += 1
            
    # 5. Save the updated data back to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print("-" * 30)
    print(f"✅ Fixed {fixed_height_count} missing heights.")
    print(f"✅ Fixed {fixed_weight_count} missing weights.")
    print(f"✅ Fixed {fixed_foot_count} missing preferred feet.")
    print(f"💾 Cleaned data saved successfully to '{output_file}'!")

# Run the script
if __name__ == "__main__":
    # Ensure this points to the exact location of your JSON file
    INPUT_JSON = "Date - meciuri/players (1).json" 
    OUTPUT_JSON = "players_fixed.json"
    
    fix_missing_physical_data(INPUT_JSON, OUTPUT_JSON)