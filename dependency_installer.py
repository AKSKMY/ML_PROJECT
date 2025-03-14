"""
Script to install required dependencies for the Motorcycle Price Prediction project
"""
import sys
import subprocess
import importlib
import os

def check_module(module_name):
    """Check if a module is available for import"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

def main():
    # Define required packages for motorcycle price prediction
    required_packages = {
        "scikit-learn": "sklearn",         # For SVM and other ML models
        "numpy": "numpy",                  # For numerical operations
        "pandas": "pandas",                # For data manipulation
        "matplotlib": "matplotlib",        # For visualization
        "seaborn": "seaborn",              # For enhanced visualization
        "joblib": "joblib",                # For saving/loading models
        "flask": "flask",                  # For web application
        "xgboost": "xgboost",              # For XGBoost model
        "lightgbm": "lightgbm",            # For LightGBM model
        "openpyxl": "openpyxl",            # For Excel file handling
        "requests": "requests",            # For web scraping
        "beautifulsoup4": "bs4"            # For web scraping
    }
    
    print("Checking for required dependencies for motorcycle price prediction...")
    missing_packages = []
    
    # Check which packages are missing
    for package_name, module_name in required_packages.items():
        if not check_module(module_name):
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        print(f"The following packages need to be installed: {', '.join(missing_packages)}")
        
        # Ask for confirmation
        if input("Do you want to install them now? (y/n): ").lower() != 'y':
            print("Installation canceled. Note that the application may not work correctly without these packages.")
            return
        
        for package in missing_packages:
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
    else:
        print("✅ All required packages are already installed!")
    
    print("\nDependency check and installation complete.")
    
    # Ask if the user wants to run the application
    if input("Do you want to run the application now? (y/n): ").lower() == 'y':
        # Get the absolute path to the main project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the Flask application
        flask_app_path = os.path.join(project_dir, "FLASKAPP", "app_v2.py")
        
        if not os.path.exists(flask_app_path):
            print(f"⚠️ Application file not found at {flask_app_path}")
            flask_app_path = os.path.join(project_dir, "app_v2.py")
            if not os.path.exists(flask_app_path):
                print(f"⚠️ Application file also not found at {flask_app_path}")
                app_files = []
                for root, dirs, files in os.walk(project_dir):
                    for file in files:
                        if file.startswith("app") and file.endswith(".py"):
                            app_files.append(os.path.join(root, file))
                
                if app_files:
                    print("Found potential application files:")
                    for i, file in enumerate(app_files):
                        print(f"{i+1}. {file}")
                    choice = input("Enter the number of the file you want to run (or 'n' to cancel): ")
                    if choice.lower() == 'n':
                        return
                    try:
                        index = int(choice) - 1
                        if 0 <= index < len(app_files):
                            flask_app_path = app_files[index]
                        else:
                            print("Invalid choice. Exiting.")
                            return
                    except ValueError:
                        print("Invalid input. Exiting.")
                        return
                else:
                    print("No application files found. Please run the application manually.")
                    return
            
        # Set PYTHONPATH to include the project directory
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{project_dir}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = project_dir
        
        print(f"Starting Flask application: {flask_app_path}")
        try:
            # Run the Flask application using python
            subprocess.run([sys.executable, flask_app_path], env=env)
        except Exception as e:
            print(f"Error running Flask application: {e}")

if __name__ == "__main__":
    main()