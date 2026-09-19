#%%
import pandas as pd
import numpy as np

np.random.seed(10)
n_orders = 500
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')

products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Webcam', 'Speaker', 'Tablet']
categories = {'Laptop': 'Electronics', 'Mouse': 'Accessories', 'Keyboard': 'Accessories',
              'Monitor': 'Electronics', 'Headphones': 'Accessories', 'Webcam': 'Accessories',
              'Speaker': 'Electronics', 'Tablet': 'Electronics'}
prices = {'Laptop': 500, 'Mouse': 15, 'Keyboard': 25, 'Monitor': 150,
          'Headphones': 40, 'Webcam': 35, 'Speaker': 60, 'Tablet': 300}
cities = ['عمان', 'إربد', 'الزرقاء', 'العقبة']

orders = pd.DataFrame({
    'order_date': np.random.choice(dates, n_orders),
    'product': np.random.choice(products, n_orders),
    'quantity': np.random.randint(1, 5, n_orders),
    'city': np.random.choice(cities, n_orders, p=[0.5, 0.2, 0.2, 0.1]),
    'customer_id': np.random.randint(1, 101, n_orders)
})
orders['category'] = orders['product'].map(categories)
orders['unit_price'] = orders['product'].map(prices)
orders['revenue'] = orders['quantity'] * orders['unit_price']
orders = orders.sort_values('order_date').reset_index(drop=True)
orders.info()
#============================
# Part # 1
#============================
#%%
# العميل الأول اشترى مرة واحدة فقط منذ فترة طويلة وبمبلغ كبير،
# بينما العميل الثاني يشتري بشكل متكرر وبمبالغ صغيرة.
# بالنسبة لمحلل البيانات، الفرق مهم لأن قيمة العميل لا تعتمد فقط على
# المبلغ الذي أنفقه، وإنما أيضًا على حداثة آخر شراء وتكرار عمليات الشراء.

# RFM يساعدنا على فهم سلوك العميل من خلال:
# Recency: متى كانت آخر عملية شراء؟
# Frequency: كم مرة اشترى؟
# Monetary: كم أنفق إجمالًا؟
#
# وبالتالي يمكننا اختيار إجراء تسويقي مناسب لكل نوع من العملاء.

print("""
RFM يساعدنا على فهم سلوك العملاء من خلال ثلاثة أبعاد:
Recency و Frequency و Monetary.

فالعميل الذي اشترى مرة واحدة فقط منذ فترة طويلة وبمبلغ كبير
يختلف عن العميل الذي يشتري بشكل متكرر حتى لو كانت قيمة كل عملية شراء صغيرة.

لذلك من المهم كمحلل بيانات ألا أعتمد فقط على إجمالي المبلغ،
بل أنظر أيضًا إلى مدى حداثة الشراء وعدد مرات الشراء.
وبناءً على ذلك يمكن اختيار استراتيجية تسويقية مناسبة لكل مجموعة من العملاء.

أمثلة على قرارات تسويقية:

1-
عميل لديه Recency مرتفعة و Frequency مرتفعة:
هذا يعني أنه كان يشتري بشكل متكرر، لكن آخر عملية شراء له كانت منذ فترة طويلة.
يمكن إرسال عرض أو خصم شخصي له بهدف استعادته وتشجيعه على الشراء مرة أخرى.

2-
عميل لديه Recency منخفضة و Frequency منخفضة:
هذا يعني أنه اشترى مؤخرًا، لكنه لم يشترِ عددًا كبيرًا من المرات حتى الآن.
يمكن إرسال توصيات لمنتجات مرتبطة بآخر عملية شراء أو عرض تشجيعي
لتحفيزه على تكرار الشراء.
""")
#============================
# Part # 2
#============================
analysis_date = orders['order_date'].max()
last_purchase = orders.groupby('customer_id')['order_date'].max()

recency = (analysis_date - last_purchase).dt.days
frequency = orders.groupby('customer_id')['customer_id'].count()
monetary = orders.groupby('customer_id')['revenue'].sum()

rfm=pd.DataFrame({'recency':recency,'frequency':frequency,'monetary':monetary})
rfm = rfm.reset_index()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency','frequency','monetary']])
rfm_scaled

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
for k in range(2,8):
    Kmeans = KMeans(n_clusters=k , random_state= 42 , n_init= 10)
    Kmeans.fit(rfm_scaled)
    inertia.append(Kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 8), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

# Based on the Elbow Method, k = 4 appears to be the most suitable choice.

best_k = 4
Kmeans = KMeans(n_clusters= best_k , random_state= 42 , n_init= 10)

rfm['segment'] = Kmeans.fit_predict(rfm_scaled) 

print("RFM table with customer segments:")
print(rfm)


#============================
# Part # 3
#============================

segment_summary = rfm.groupby('segment')[['recency','frequency','monetary']].mean()

segment_names = {0 : 'At Risk' , 1 : "Regular" , 2 : 'New Customer' , 3 : 'Champions'}
segment_summary['segment_name'] = segment_summary.index.map(segment_names)
rfm['segment_name'] = rfm['segment'].map(segment_names)


# Champions : هم عملاء مميزون انفاقهم مرتفع , بيشتروا بشكل متكرر , و اخر عملية شراء كانت قريبة
# الاجراء التسويقي : عروض حصرية خاصه فيهم منشان احافظ عليهم كعملاء مميزين

# At Risk : عملاء بخطر الفقدان و همه عملاء الهم مدة طويلة ما اشترو لكن عمليات شراءهم كانت بمبالغ كبيرة و متكررة
# اجراء تسويقي : عرض شخصي او خصم لاعادتهم

# New Customers : اشترو موخرا بس ما الهم تاريخ سابق طويل من عمليات الشراء
# عروض تشجيعيه حتى يشترو مرة تانيه او توصيات لمنتجات مناسبة خاصة فيهم بناءا على عملية الشراء السابقة متلا

# Regular : عملاء عاديين قيمهم متوسطة جميعها
# الاجراء التسويقي : عروض دورية لزيادة قيمة المشتريات

print("""
الجدول التالي يوضح متوسط قيم Recency و Frequency و Monetary لكل Segment،
ومن خلاله نحدد خصائص كل مجموعة من العملاء ونختار الاسم المناسب لها.
""")

print(segment_summary)

print("""
0 - At Risk:
عملاء بخطر الفقدان، وهم عملاء مرّ وقت طويل منذ آخر عملية شراء لهم،
لكن عمليات الشراء لديهم كانت بمبالغ كبيرة ومتكررة.
الإجراء التسويقي: عرض شخصي أو خصم لإعادتهم وتشجيعهم على الشراء مرة أخرى.


1 - Regular:
عملاء عاديين، وقيم الـ Recency و Frequency و Monetary لديهم متوسطة.
الإجراء التسويقي: عروض دورية لزيادة تكرار الشراء أو قيمة المشتريات.


2 - New Customer:
اشتروا مؤخرًا، لكن ليس لديهم تاريخ طويل من عمليات الشراء.
الإجراء التسويقي: عروض تشجيعية حتى يشتروا مرة أخرى،
أو توصيات لمنتجات مناسبة لهم بناءً على عملية الشراء السابقة.


3 - Champions:
هم عملاء مميزون، إنفاقهم مرتفع، ويشترون بشكل متكرر،
وآخر عملية شراء لهم كانت قريبة.
الإجراء التسويقي: عروض حصرية خاصة بهم ومكافآت للحفاظ عليهم كعملاء مميزين.
""")


#============================
# Part # 4
#============================

for segment in sorted(rfm['segment'].unique()):
    data = rfm[rfm['segment'] == segment]
    
    plt.scatter(
        x=data['frequency'],
        y=data['monetary'],
        label=segment_names[segment]
    )

plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Customer Segments: Frequency vs Monetary')
plt.legend()
plt.savefig('customer_segments_frequency_vs_monetary.png',dpi=300,bbox_inches='tight')
plt.show()

segment_counts = rfm['segment'].value_counts().sort_index()
segment_table = pd.DataFrame({'customer_count': segment_counts,'percentage': segment_counts / len(rfm) * 100})
segment_table = segment_table.reset_index()
segment_table['segment_name'] = segment_table['segment'].map(segment_names)
segment_table = segment_table[['segment', 'segment_name', 'customer_count', 'percentage']]
segment_table['percentage'] = segment_table['percentage'].round(2)

print("""
الجدول التالي يوضح عدد العملاء ونسبة كل Segment من إجمالي العملاء.
""")
print(segment_table)


revenue_by_segment = rfm.groupby('segment')['monetary'].sum()

revenue_percentage = (revenue_by_segment / rfm['monetary'].sum() * 100).round(2)

revenue_table = pd.DataFrame({'total_revenue': revenue_by_segment,'revenue_percentage': revenue_percentage})
revenue_table = revenue_table.reset_index()
revenue_table['segment_name'] = revenue_table['segment'].map(segment_names)
revenue_table = revenue_table[['segment', 'segment_name', 'total_revenue', 'revenue_percentage']]
revenue_table = revenue_table.sort_values('revenue_percentage',ascending=False)

print("""
الجدول التالي يوضح إجمالي الإيرادات ونسبة مساهمة كل Segment من إجمالي الإيرادات،
مما يساعدنا على معرفة أي مجموعات العملاء تساهم بشكل أكبر في الإيرادات.
""")
print(revenue_table)
top_segment = revenue_table.iloc[0]

print(f"""
أعلى Segment من حيث الإيرادات هو:
{top_segment['segment_name']}

إجمالي الإيرادات:
{top_segment['total_revenue']:.0f}

نسبة مساهمته من إجمالي الإيرادات:
{top_segment['revenue_percentage']:.2f}%
""")

# ==========================================
# Part 4 - Final Segmentation Report
# ==========================================

print("""
==========================================
Final Segmentation Report
==========================================

يهدف هذا التحليل إلى تقسيم العملاء إلى مجموعات مختلفة بناءً على
سلوكهم الشرائي باستخدام مؤشرات Recency و Frequency و Monetary
وتطبيق خوارزمية K-Means.

أظهر التحليل وجود أربع مجموعات رئيسية من العملاء. تمثل مجموعة
Regular أكبر عدد من العملاء، حيث تضم 41.41% من إجمالي العملاء،
بينما تمثل Champions العملاء ذوي أعلى مستوى من التكرار والإنفاق.
أما At Risk فتضم العملاء الذين مر وقت طويل منذ آخر عملية شراء لهم،
في حين تضم New Customer عملاء لديهم عدد محدود نسبيًا من عمليات الشراء
وإنفاق منخفض مقارنة بالمجموعات الأعلى قيمة.

من ناحية الإيرادات، كانت مجموعة Regular هي الأعلى مساهمة في إجمالي
الإيرادات، حيث حققت 86,445 وبنسبة 47.46% من إجمالي الإيرادات.
لذلك يمكن التركيز على الحفاظ على عملاء Regular وتشجيعهم على زيادة
قيمة مشترياتهم، مع استخدام عروض إعادة التنشيط لعملاء At Risk،
وتشجيع New Customer على تكرار الشراء، والحفاظ على Champions من خلال
المكافآت والعروض الحصرية.
""")