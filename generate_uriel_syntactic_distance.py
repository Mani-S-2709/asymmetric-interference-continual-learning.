import pandas as pd
from urielplus import urielplus

# Initialize URIEL+
u = urielplus.URIELPlus()

# Languages in the Phase 2 pilot
languages = ["eng", "hin", "spa", "deu", "swa"]

# Calculate the URIEL+ syntactic distance matrix
distance_matrix = u.new_distance("syntactic", languages)

# Convert to a labeled DataFrame
df = pd.DataFrame(
    distance_matrix,
    index=languages,
    columns=languages
)

# Save the CSV
output_path = "uriel_syntactic_distance.csv"
df.to_csv(output_path, index=True)

print(df)
print(f"\nSaved: {output_path}")
print("Matrix shape:", df.shape)