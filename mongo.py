from pymongo import MongoClient
from decouple import config

# Connect to MongoDB Atlas
client = MongoClient(config("MONGO_URI"))
db = client[config("DB_NAME")]
collection = db[config("COLLECTION_NAME")]

# Query the database and retrieve data
# Example: Find all documents in the collection
cursor = collection.find({})

# Iterate through the cursor to access the documents
for document in cursor:
    for i, v in document.items():
        print(f"{i}: {v}")

# Close the MongoDB connection when done
client.close()
