import pandas as pd

print("Loading dataset...")
df = pd.read_csv('911.csv/911.csv')

high_risk_keywords = ['CARDIAC', 'RESPIRATORY', 'HEAD INJURY', 'SEIZURES', 'UNCONSCIOUS', 'OVERDOSE', 'CVA/STROKE', 'SHOOTING', 'STABBING', 'CHOKING', 'MATERNITY']
medium_risk_keywords = ['VEHICLE ACCIDENT', 'FALL VICTIM', 'SUBJECT IN PAIN', 'ABDOMINAL PAINS', 'HEMORRHAGE', 'ASSAULT', 'FIRE ALARM', 'BURN VICTIM', 'ALLERGIC REACTION', 'LACERATIONS']

def get_risk_label(title):
    title_upper = title.upper()
    for kw in high_risk_keywords:
        if kw in title_upper:
            return 2 # HIGH
    for kw in medium_risk_keywords:
        if kw in title_upper:
            return 1 # MEDIUM
    return 0 # LOW

print("Applying risk labels...")
df['risk'] = df['title'].apply(get_risk_label)
df['text'] = df['title']

print("Sampling data...")
df_high = df[df['risk'] == 2].sample(n=200, random_state=42)
df_medium = df[df['risk'] == 1].sample(n=200, random_state=42)
df_low = df[df['risk'] == 0].sample(n=200, random_state=42)

df_sampled = pd.concat([df_high, df_medium, df_low]).sample(frac=1, random_state=42).reset_index(drop=True)
df_sampled = df_sampled[['text', 'risk']]
df_sampled.rename(columns={'risk': 'label'}, inplace=True)

df_sampled.to_csv('training_data.csv', index=False)
print("Saved training_data.csv with", len(df_sampled), "rows.")
