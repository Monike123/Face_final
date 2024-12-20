import os
import cloudinary
import cloudinary.uploader
from flask import Flask, request, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor
import face_recognition
from PIL import Image
import numpy as np

app = Flask(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name="dgnoyjwax",  # Replace with your Cloudinary Cloud Name
    api_key="678962427316572",        # Replace with your Cloudinary API Key
    api_secret="nkvt1twiZzv2RvYugi8ASv7h_cw"   # Replace with your Cloudinary API Secret
)

# Aiven PostgreSQL Database configuration
DB_HOST = 'face-opop.g.aivencloud.com'
DB_PORT = '21703'
DB_USER = 'avnadmin'
DB_PASSWORD = 'AVNS_X20xXPQrX-lExpXvOyo'
DB_NAME = 'defaultdb'

@app.route('/store_image', methods=['POST'])
def store_image():
    try:
        # Check if the image file is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Retrieve the image from the request
        file = request.files['image']

        # Upload the image to Cloudinary
        cloudinary_response = cloudinary.uploader.upload(file)
        image_url = cloudinary_response['secure_url']  # URL of the uploaded image

        # Process the image for facial encoding
        img = Image.open(file.stream)
        img_array = np.array(img)

        # Get facial encodings
        encodings = face_recognition.face_encodings(img_array)
        if not encodings:
            return jsonify({'error': 'No face detected in the image'}), 400

        # Convert encodings to binary for database storage
        encoding_binary = np.array(encodings[0], dtype=np.float32).tobytes()

        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME
        )
        cur = conn.cursor()

        # Insert image URL and encoding into the database
        cur.execute(
            "INSERT INTO faces (filename, embedding) VALUES (%s, %s) RETURNING id",
            (image_url, psycopg2.Binary(encoding_binary))
        )
        conn.commit()

        # Retrieve the ID of the inserted record
        image_id = cur.fetchone()[0]

        # Close database connection
        cur.close()
        conn.close()

        return jsonify({'message': 'Image stored successfully', 'image_id': image_id, 'image_url': image_url}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/verify_image', methods=['POST'])
def verify_image():
    try:
        # Check if the image file is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Retrieve the image from the request
        file = request.files['image']
        img = Image.open(file.stream)

        # Convert the image to RGB (if not already)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img)

        # Get facial encodings
        query_encodings = face_recognition.face_encodings(img_array)
        if not query_encodings:
            return jsonify({'error': 'No face detected in the image'}), 400

        query_embedding = query_encodings[0]

        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Retrieve embeddings from the database
        cur.execute("SELECT id, filename, embedding FROM faces")
        rows = cur.fetchall()

        best_similarity = -1
        matched_image = None

        # Compare with database encodings
        for row in rows:
            db_embedding = np.frombuffer(row['embedding'], dtype=np.float32)

            # Compute face distance
            distance = face_recognition.face_distance([db_embedding], query_embedding)[0]
            similarity = max(0, 1 - distance)  # Convert distance to similarity

            if similarity > best_similarity:
                best_similarity = similarity
                matched_image = row['filename']

        cur.close()
        conn.close()

        # Response based on similarity
        if matched_image and best_similarity >= 0.5:
            return jsonify({
                'message': 'Match found',
                'matched_image': matched_image,
                'similarity': best_similarity
            }), 200
        else:
            return jsonify({'message': 'No match found', 'similarity': best_similarity}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
