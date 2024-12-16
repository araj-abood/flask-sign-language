from neural_hand_detector import NeuralHandSignDetector
from camera_utils import DataCollector, PredictionTester

def main():
    detector = NeuralHandSignDetector()
    data_collector = DataCollector()
    prediction_tester = PredictionTester()
    
    # Define your Arabic signs
    signs = ["علامة السلام"]  
    samples_needed = 30
    
    while True:
        print("\n=== Hand Sign Recognition System ===")
        print("1. Collect training data (with timer)")
        print("2. Train model")
        print("3. Test predictions")
        print("4. Save model")
        print("5. Load model")
        print("6. Quit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        try:
            if choice == '1':
                data_collector.collect_data_with_timer(detector, signs, samples_needed)
            elif choice == '2':
                print("\nTraining model...")
                detector.train_model(epochs=100)
                print("Training completed!")
            elif choice == '3':
                if not hasattr(detector, 'model') or detector.model is None:
                    print("\nError: No trained model found. Please train or load a model first.")
                    continue
                prediction_tester.test_predictions(detector)
            elif choice == '4':
                detector.save_model('neural_hand_signs.pth')
                print("\nModel saved to 'neural_hand_signs.pth'")
            elif choice == '5':
                try:
                    detector.load_model('neural_hand_signs.pth')
                    print("\nModel loaded successfully!")
                except FileNotFoundError:
                    print("\nError: No saved model found!")
                except Exception as e:
                    print(f"\nError loading model: {e}")
            elif choice == '6':
                break
            else:
                print("\nInvalid choice! Please try again.")
                
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()