from flask import Flask, request, jsonify, render_template, send_from_directory
import face_recognition
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Database connection parameters
DB_HOST = 'face-opop.g.aivencloud.com'
DB_PORT = '21703'
DB_USER = 'avnadmin'
DB_PASSWORD = 'AVNS_X20xXPQrX-lExpXvOyo'
DB_NAME = 'defaultdb'

# Directory to save and serve images
IMAGE_DIR = 'images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


@app.route('/store_image', methods=['POST'])
def store_image():
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME
        )
        cur = conn.cursor()

        # Read image from POST request
        file = request.files['image']
        img = Image.open(file.stream)
        img = np.array(img)  # Convert PIL Image to numpy array

        # Save the image
        image_path = os.path.join(IMAGE_DIR, file.filename)
        Image.fromarray(img).save(image_path)

        # Use face_recognition to get encodings
        encodings = face_recognition.face_encodings(img)

        if len(encodings) == 0:
            return jsonify({'error': 'No face detected in the image'}), 400

        encoding = encodings[0]  # Take the first face encoding found
        encoding_binary = np.array(encoding, dtype=np.float32).tobytes()

        # Insert image and encoding into the database
        cur.execute("INSERT INTO faces (filename, embedding) VALUES (%s, %s) RETURNING id",
                    (file.filename, psycopg2.Binary(encoding_binary)))
        conn.commit()
        image_id = cur.fetchone()[0]  # Get the ID of the inserted image
        cur.close()
        conn.close()

        return jsonify({'message': 'Image stored successfully', 'image_id': image_id}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/images/<filename>')
def serve_image(filename):
    image_path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(image_path):
        return send_from_directory(IMAGE_DIR, filename)
    else:
        return jsonify({'error': 'Image not found'}), 404


@app.route('/verify_image', methods=['POST'])
def verify_image():
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Read image from POST request
        file = request.files['image']
        img = Image.open(file.stream)

        # Convert the image to RGB (if not already)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = np.array(img)  # Convert PIL Image to numpy array

        # Use face_recognition to get encodings
        query_encodings = face_recognition.face_encodings(img)

        if len(query_encodings) == 0:
            print(f"No face detected in the image: {file.filename}")
            return jsonify({'error': 'No face detected in the image'}), 400

        query_embedding = query_encodings[0]  # Get the first encoding

        # Retrieve encodings from the database
        cur.execute("SELECT id, filename, embedding FROM faces")
        rows = cur.fetchall()

        best_similarity = -1  # Initialize to -1 to ensure the first similarity is better
        matched_image = None

        for row in rows:
            db_embedding = np.frombuffer(row['embedding'], dtype=np.float32)

            if len(query_embedding) != len(db_embedding):
                print(f"Skipping due to dimension mismatch: query {len(query_embedding)}, db {len(db_embedding)}")
                continue

            distance = face_recognition.face_distance([db_embedding], query_embedding)[0]
            similarity = max(0, 1 - distance)  # Convert distance to similarity

            print(f"Comparing with {row['filename']}, similarity: {similarity:.2f}")

            if similarity > best_similarity:
                best_similarity = similarity
                matched_image = row['filename']

        cur.close()
        conn.close()

        # Determine similarity percentage
        similarity_percentage = best_similarity if best_similarity >= 0 else 0  # Convert to percentage
        print(f"Final similarity percentage: {similarity_percentage:.2f}")

        # Response based on best similarity found
        if matched_image and best_similarity >= 0.5:  # Adjusted threshold
            return jsonify({'message': 'Match found', 'matched_image': matched_image, 'similarity': similarity_percentage}), 200
        else:
            return jsonify({'message': 'No match found', 'similarity': similarity_percentage}), 404  # Change response code to 200

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug log for the error
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
