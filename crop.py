import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Imbalance uchun
import plotly.express as px
import plotly.graph_objects as go

# Dataset - yuklangan Excel fayl (nam_tup.xlsx) dan o'qish
# Eslatma: Real muhitda fayl nomi va yo'li to'g'ri bo'lishi kerak. Bu yerda "nam_tup.xlsx" deb qo'yilgan.
DATA_URL= "https://raw.githubusercontent.com/abroraxatov1/soil_dataset/refs/heads/main/nam_tup.csv"

def parse_layer(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if 'sm' in value or 'cm' in value:
        value = value.replace(' sm', '').replace(' cm', '')
    if '-' in value:
        parts = value.split('-')
        if len(parts) == 2:
            try:
                start = float(parts[0].strip())
                end = float(parts[1].strip())
                return (start + end) / 2
            except ValueError:
                pass
    try:
        return float(value)
    except ValueError:
        return np.nan

@st.cache_data
def load_and_preprocess():
    # Excel faylni o'qish, birinchi sheet (Лист1)
    df = pd.read_csv(DATA_URL)
    
    # Ustun nomlarini to'g'rilash (agar kerak bo'lsa, lekin berilgan ma'lumotga asosan to'g'ri)
    expected_columns = [
        'Namuna', 'Qatlam (sm)', 'Mexanik tarkib', 'DNS (%)', 'Tuproq zichligi (g/cm³)',
        'pH', 'EC (mS/cm)', 'N (mg/kg)', 'P (mg/kg)', 'K (mg/kg)', 'Gumus (%)',
        'Mg (mg/kg)', 'S (mg/kg)', 'Zn (mg/kg)', 'Mn (mg/kg)', 'B (mg/kg)',
        'Fe (mg/kg)', 'Cu (mg/kg)', 'Mikroorganizmlar(CFU/g)', 'Ekin'
    ]
    
    # Mavjud ustunlarni tekshirish va moslashtirish
    df.columns = [col.strip() for col in df.columns]  # Bo'shliqlarni olib tashlash
    
    # Ekin tozalash
    df['Ekin'] = df['Ekin'].fillna(df['Ekin'].mode()[0] if not df['Ekin'].mode().empty else 'Unknown').astype(str)
    
    # Qatlam parsing
    if 'Qatlam (sm)' in df.columns:
        df['Qatlam (sm)'] = df['Qatlam (sm)'].apply(parse_layer)
        df['Qatlam (sm)'] = pd.to_numeric(df['Qatlam (sm)'], errors='coerce').fillna(df['Qatlam (sm)'].mean())
    
    # NaN to'ldirish
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if col != 'Ekin' and col != 'Namuna':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown').astype(str)
    
    # Encode
    le_dict = {}
    for col in categorical_cols:
        if col != 'Ekin' and col != 'Namuna':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
    
    le_crop = LabelEncoder()
    df['ekin_encoded'] = le_crop.fit_transform(df['Ekin'])
    
    # Ekin bo'yicha o'rtacha qiymatlar (radar uchun)
    crop_averages = df.groupby('Ekin')[numeric_cols].mean()
    
    return df, le_dict, le_crop, crop_averages

@st.cache_resource
def train_model(df):
    X = df.drop(['Namuna', 'Ekin', 'ekin_encoded'], axis=1, errors='ignore')
    y = df['ekin_encoded']
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE bilan imbalance hal qilish (datasetdagi imbalance ni bartaraf etish uchun)
    smote = SMOTE(random_state=42, k_neighbors=5)  # k_neighbors ni sozlash overfitting oldini olish uchun
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Model parametrlari: overfitting oldini olish uchun max_depth, min_samples va boshqalarni cheklash
    rf = RandomForestClassifier(
        n_estimators=150,  # Ko'proq estimatorlar qo'shish
        max_depth=12,      # Chuqurlikni oshirish, lekin cheklash
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',  # Feature larni cheklash
        random_state=42,
        class_weight='balanced'  # Class imbalance uchun
    )
    rf.fit(X_train_res, y_train_res)
    
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    cv_scores = cross_val_score(rf, X_train_res, y_train_res, cv=5, scoring='accuracy')
    
    report = classification_report(y_test, rf.predict(X_test), target_names=le_crop.classes_, output_dict=True)
    
    # Feature importances
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return rf, X.columns.tolist(), test_acc, cv_scores.mean(), train_acc, report, feature_importances

# App
st.set_page_config(page_title="Ekin tavfsiyasi", layout="wide")
st.title("Sharof Rashidov nomidagi Samarqand davlat universiteti “Sun’iy intellekt” labaratoriyasi tomonidan ishlab chiqilgan tuproq ma'lumtotlari aosida ekin tavfsiya qilish sun’iy intellekt modeli ")

# Yuklash
with st.spinner("Model ishga tushirilmoqda..."):
    df, le_dict, le_crop, crop_averages = load_and_preprocess()
    rf, feature_names, test_acc, cv_acc, train_acc, report, feature_importances = train_model(df)

# Sidebar inputlar
st.sidebar.header("Tuproq xusuiyatlarini kiriting")

# Qatlam (sm) uchun options - datasetdan olingan umumiy qiymatlar asosida
layer_options = ['0-20', '20-40', '40-60', '0-30', '60-90']  # Datasetdan olingan misollar
selected_layer = st.sidebar.selectbox("Qatlam (sm)", layer_options)
layer_value = (float(selected_layer.split('-')[0]) + float(selected_layer.split('-')[1])) / 2

# Mexanik tarkib encoded
mech_comp_encoded = 0
if 'Mexanik tarkib' in le_dict:
    mech_classes = le_dict['Mexanik tarkib'].classes_
    mech_comp_str = st.sidebar.selectbox("Mexanik tarkib", mech_classes)
    mech_comp_encoded = le_dict['Mexanik tarkib'].transform([mech_comp_str])[0]

# Sliderlar (dataset qiymatlari asosida min/max sozlangan, float)
dns = st.sidebar.slider("DNS (%)", 10.0, 40.0, 25.0)
density = st.sidebar.slider("Tuproq zichligi (g/cm³)", 1.0, 1.8, 1.4)
ph = st.sidebar.slider("pH", 4.5, 8.5, 6.5)
ec = st.sidebar.slider("EC (mS/cm)", 0.1, 5.0, 1.0)
nitrogen = st.sidebar.slider("N (mg/kg)", 10.0, 200.0, 100.0)
phosphorus = st.sidebar.slider("P (mg/kg)", 5.0, 100.0, 50.0)
potassium = st.sidebar.slider("K (mg/kg)", 50.0, 500.0, 250.0)
humus = st.sidebar.slider("Gumus (%)", 0.5, 5.0, 2.0)
mg = st.sidebar.slider("Mg (mg/kg)", 20.0, 500.0, 150.0)  # Datasetda yuqori qiymatlar bor
s = st.sidebar.slider("S (mg/kg)", 5.0, 500.0, 50.0)    # Datasetda yuqori
zn = st.sidebar.slider("Zn (mg/kg)", 0.5, 10.0, 5.0)
mn = st.sidebar.slider("Mn (mg/kg)", 1.0, 60.0, 25.0)
b = st.sidebar.slider("B (mg/kg)", 0.1, 5.0, 2.0)
fe = st.sidebar.slider("Fe (mg/kg)", 10.0, 400.0, 100.0)  # Datasetda yuqori
cu = st.sidebar.slider("Cu (mg/kg)", 0.5, 10.0, 5.0)
microorg = st.sidebar.slider("Mikroorganizmlar (CFU/g)", 1e5, 1e10, 1e8)  # Datasetda katta qiymatlar

# Asosiy oynada kiritilgan qiymatlarni ko'rsatish (jadval ko'rinishida)
st.subheader("Kiritilgan tuproq qiymatlari")
input_summary = {
    "Qatlam (sm)": layer_value,
    "Mexanik tarkib": mech_comp_str,
    "DNS (%)": dns,
    "Tuproq zichligi (g/cm³)": density,
    "pH": ph,
    "EC (mS/cm)": ec,
    "N (mg/kg)": nitrogen,
    "P (mg/kg)": phosphorus,
    "K (mg/kg)": potassium,
    "Gumus (%)": humus,
    "Mg (mg/kg)": mg,
    "S (mg/kg)": s,
    "Zn (mg/kg)": zn,
    "Mn (mg/kg)": mn,
    "B (mg/kg)": b,
    "Fe (mg/kg)": fe,
    "Cu (mg/kg)": cu,
    "Mikroorganizmlar (CFU/g)": microorg
}
st.table(pd.DataFrame(list(input_summary.items()), columns=["Xususiyat", "Qiymat"]))

# Bashorat
if st.sidebar.button("Kiritish", type="primary"):
    input_dict = {
        'Qatlam (sm)': layer_value,
        'Mexanik tarkib': mech_comp_encoded,
        'DNS (%)': dns,
        'Tuproq zichligi (g/cm³)': density,
        'pH': ph,
        'EC (mS/cm)': ec,
        'N (mg/kg)': nitrogen,
        'P (mg/kg)': phosphorus,
        'K (mg/kg)': potassium,
        'Gumus (%)': humus,
        'Mg (mg/kg)': mg,
        'S (mg/kg)': s,
        'Zn (mg/kg)': zn,
        'Mn (mg/kg)': mn,
        'B (mg/kg)': b,
        'Fe (mg/kg)': fe,
        'Cu (mg/kg)': cu,
        'Mikroorganizmlar(CFU/g)': microorg
    }
    
    input_data = [input_dict.get(col, 0.0) for col in feature_names]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Har ekin uchun alohida moslik (0-100%, jami 100% emas)
    probs = rf.predict_proba(input_df)[0]
    crop_probs = {le_crop.classes_[i]: min(probs[i] * 100, 100) for i in range(len(le_crop.classes_))}
    
    df_probs = pd.DataFrame(list(crop_probs.items()), columns=['Ekin', 'Moslik (%)']).sort_values('Moslik (%)', ascending=False)
    st.subheader("Ekinlarning Moslik Foizlari (0-100%)")
    st.table(df_probs)
    
    fig = px.bar(df_probs, x='Ekin', y='Moslik (%)', title="Ekin Mosligi (0-100% Diagrammasi)", color='Moslik (%)', text='Moslik (%)')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig)
    
    best_crop = df_probs.iloc[0]['Ekin']
    st.success(f"**Eng mos: {best_crop} ({df_probs.iloc[0]['Moslik (%)']:.1f}%)**")
    
    # Eng mos ekin uchun rasm va radar grafik yonma-yon, jadval tagida
    col1, col2 = st.columns(2)
    
    with col1:
        # Ekin rasmi kichikroq
        crop_images = {
            "Bug'doy": "https://i0.wp.com/razzanj.com/wp-content/uploads/2016/07/nature-landscape-nature-landscape-hd-image-download-wheat-farm-hd-wallpaper-notebook-background-wheat-farmers-wheat-farming-process-wheat-farming-in-kenya.jpg?ssl=1",
            "Kartoshka": "https://www.isaaa.org/kc/cropbiotechupdate/files/images/3172020111359PM.jpg",
            "Loviya": "https://cdn.britannica.com/24/122524-050-4593E7D1/Green-beans.jpg",
            "Qanampir": "https://media.istockphoto.com/id/1359749116/photo/red-chili-peppers-in-vegetable-garden.jpg?s=612x612&w=0&k=20&c=8Kz7TcxH0Cl9A2tWzMkeWVoIFD71LdDDchQWpoPyQzE=",
            "Makkajo'xori": "https://www.aces.edu/wp-content/uploads/2018/08/shutterstock_-Zeljko-Radojko_field-corn.jpg",
            "Sabzi": "https://ogden_images.s3.amazonaws.com/www.motherearthnews.com/images/2022/02/11110505/growing-carrots.jpg",
            "Paxta": "https://cdn.pixabay.com/photo/2014/02/13/12/56/cotton-crop-265312_1280.jpg",



            
        }
        if best_crop in crop_images:
            st.image(crop_images[best_crop], caption=f"Berilgan tuproq maydoniga eng mos ekin {best_crop}", width=600)  # Kichikroq width
        else:
            st.info(f"{best_crop} uchun rasm topilmadi. Ma'lumotlar to'plamida mavjud ekinlar: {', '.join(le_crop.classes_)}")
    
    with col2:
        # Radar grafik
        st.subheader(f"{best_crop} uchun eng muhim xususiyatlar")
        top_features = feature_importances.head(6).index.tolist()  # Eng muhim 6 ta xususiyat
        
        def normalize_series(s):
            return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s / s.max()
        
        input_values = pd.Series([input_dict.get(f, 0) for f in top_features], index=top_features)
        avg_values = crop_averages.loc[best_crop, top_features] if best_crop in crop_averages.index else pd.Series(0, index=top_features)
        
        input_norm = normalize_series(input_values)
        avg_norm = normalize_series(avg_values)
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=input_norm.tolist() + [input_norm.iloc[0]],
            theta=top_features + [top_features[0]],
            fill='toself',
            name='Kiritilgan qiymatlar'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_norm.tolist() + [avg_norm.iloc[0]],
            theta=top_features + [top_features[0]],
            fill='toself',
            name=f"{best_crop} O'rtacha"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )
        st.plotly_chart(fig_radar)

# Pastda ma'lumotlar
with st.expander("Ma'lumotlar to'plami va model ma'lumotlari"):
    st.metric("Ma'lumotlar to'plami hajmi", f"{df.shape[0]} qator")
    st.metric("Train aniqligi", f"{train_acc:.2%}")
    st.metric("Test aniqligi", f"{test_acc:.2%}")
    st.metric("Cross-Validation aniqligi", f"{cv_acc:.2%}")
    
    st.write("**Ma'lumotlar to'plamining o'rtacha qiymatlari (min, max, mean, std, etc.):**")
    stats = df.describe()
    st.table(stats)
    
    st.write("**Class Taqqoslovi (Imbalance Tekshirish):**")
    class_counts = df['Ekin'].value_counts()
    st.bar_chart(class_counts)

st.sidebar.markdown("---")
