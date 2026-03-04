from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

reviews = [
    ("fake", "SHOCKING secret the government does not want you to know about"),
    ("fake", "Celebrity found dead in mysterious circumstances media silent"),
    ("fake", "Scientists confirm drinking bleach cures all diseases immediately"),
    ("fake", "BREAKING aliens have landed and world leaders are hiding the truth"),
    ("fake", "Miracle cure discovered big pharma is suppressing this information"),
    ("fake", "President secretly reptilian insider leaks shocking evidence"),
    ("fake", "Vaccines contain microchips to track your every movement"),
    ("fake", "Moon landing was filmed in a Hollywood studio new proof emerges"),
    ("fake", "Drinking coffee causes instant cancer doctors warn public"),
    ("fake", "World ending tomorrow NASA hiding asteroid headed for earth"),
    ("real", "Government announces new infrastructure budget for road repairs"),
    ("real", "Scientists publish study on effects of climate change on sea levels"),
    ("real", "Local elections results announced voter turnout highest in decade"),
    ("real", "Central bank raises interest rates to control inflation"),
    ("real", "New hospital opens in the city creating 500 jobs for locals"),
    ("real", "University research finds link between sleep and memory retention"),
    ("real", "Parliament passes new data privacy law affecting tech companies"),
    ("real", "Farmers report lower crop yields due to drought this season"),
    ("real", "City council approves plan for new public transport network"),
    ("real", "Health ministry releases annual report on disease prevention"),
]

labels = [label for label, msg in reviews]
texts = [msg for label, msg in reviews]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = [1 if label == "real" else 0 for label in labels]

model1 = LogisticRegression()
model2 = MultinomialNB()
model1.fit(X, y)
model2.fit(X, y)

def predict(msg):
    x = vectorizer.transform([msg])
    result1 = model1.predict(x)[0]
    result2 = model2.predict(x)[0]
    return f"LR: {'REAL' if result1 == 1 else 'FAKE'} | NB: {'REAL' if result2 == 1 else 'FAKE'}"

while True:
    user_input = input("Enter a message: ")
    print(predict(user_input))
