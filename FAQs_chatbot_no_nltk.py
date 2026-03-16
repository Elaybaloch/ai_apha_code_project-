from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ dataset
questions = [
    "What is artificial intelligence",
    "What is machine learning",
    "What is deep learning",
    "What is natural language processing",
    "What is computer vision"
]

answers = [
    "Artificial Intelligence is the simulation of human intelligence in machines.",
    "Machine learning is a branch of AI that allows systems to learn from data.",
    "Deep learning is a subset of machine learning that uses neural networks.",
    "Natural Language Processing allows computers to understand human language.",
    "Computer vision allows machines to understand images and videos."
]

# Preprocess: lowercase only
processed_questions = [q.lower() for q in questions]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

def chatbot(user_input):
    user_input = user_input.lower()
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, X)
    index = similarity.argmax()
    return answers[index]

# Chat loop
print("FAQ Chatbot Started (type 'exit' to stop)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot stopped")
        break
    response = chatbot(user_input)
    print("Bot:", response)
