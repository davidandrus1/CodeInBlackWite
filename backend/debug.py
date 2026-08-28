from data_processor import process_data

df = process_data(
    "../Date - meciuri/players (1).json",
    "../Date - meciuri"
)

# Caută jucători sintetici după ID (încep de la 9000001)
sintetici = df[df['player_id'].astype(str).str.startswith('9')]
print(f"Sintetici în df_master: {len(sintetici)}")
print(sintetici[['original_name', 'minutes_played']].head(10))