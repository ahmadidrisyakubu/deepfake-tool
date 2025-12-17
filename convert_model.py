# convert_model.py
import tensorflow as tf
import os

def convert_to_tflite():
    print("=== TensorFlow Lite Conversion ===")
    
    # Check if original model exists
    if not os.path.exists("best_xception_model_finetuned.keras"):
        print("ERROR: Original model file not found!")
        print("Make sure 'best_xception_model_finetuned.keras' is in the same directory")
        return
    
    # Get original file size
    original_size = os.path.getsize("best_xception_model_finetuned.keras")
    print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
    
    try:
        # Step 1: Load the Keras model
        print("\n1. Loading Keras model...")
        model = tf.keras.models.load_model("best_xception_model_finetuned.keras")
        print("   ✓ Model loaded successfully")
        
        # Step 2: Create converter
        print("\n2. Creating TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional optimizations (reduces size further)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        
        # Step 3: Convert model
        print("3. Converting to TFLite format...")
        tflite_model = converter.convert()
        print("   ✓ Conversion successful")
        
        # Step 4: Save TFLite model
        print("\n4. Saving TFLite model...")
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # Check new file size
        tflite_size = os.path.getsize("model.tflite")
        print(f"   ✓ TFLite model saved as 'model.tflite'")
        print(f"   New size: {tflite_size / 1024 / 1024:.2f} MB")
        print(f"   Size reduction: {(original_size - tflite_size) / original_size * 100:.1f}%")
        
        # Step 5: Test the conversion
        print("\n5. Testing TFLite model...")
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   ✓ Input shape: {input_details[0]['shape']}")
        print(f"   ✓ Output shape: {output_details[0]['shape']}")
        print(f"   ✓ Input dtype: {input_details[0]['dtype']}")
        
        print("\n✅ Conversion completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_to_tflite()
