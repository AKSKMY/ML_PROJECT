"""
Script to install required dependencies for the Motorcycle Price Prediction project
"""
import sys
import subprocess
import importlib
import os
import platform

def check_module(module_name):
    """Check if a module is available for import"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def check_rust_installed():
    """Check if Rust/Cargo is installed"""
    try:
        result = subprocess.run(['cargo', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_package(package_name, options=None):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if options:
        cmd.extend(options)
    
    cmd.append(package_name)
    
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

def main():
    # Get Python version
    python_version = sys.version_info
    python_ver_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    print(f"Python version: {python_ver_str}")
    
    # Check if Python version is 3.11.x
    if python_version.major != 3 or python_version.minor != 11:
        print(f"❌ ERROR: This project requires Python 3.11.x, but you are using Python {python_ver_str}")
        print("Please install Python 3.11 and try again.")
        if python_version.major == 3 and python_version.minor > 11:
            print("Note: Newer versions of Python (3.12+) are not compatible with some of the dependencies used in this project.")
        elif python_version.major == 3 and python_version.minor < 11:
            print("Note: This project uses features that require Python 3.11 specifically.")
        
        # Ask if user wants to continue anyway (not recommended)
        if input("\nDo you want to continue anyway? This is NOT RECOMMENDED and may cause errors. (y/n): ").lower() != 'y':
            print("Installation canceled. Please install Python 3.11 before proceeding.")
            return
        print("\n⚠️ Continuing with an unsupported Python version. You may encounter errors!\n")
    
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
        "beautifulsoup4": "bs4",           # For web scraping
        "waitress": "waitress",            # For production server deployment
        "scipy": "scipy"                   # For statistical functions and stats module
    }
    
    # CatBoost requires special handling due to Rust dependencies
    catboost_required = False
    
    print("Checking for required dependencies for motorcycle price prediction...")
    missing_packages = []
    
    # Check which packages are missing
    for package_name, module_name in required_packages.items():
        if not check_module(module_name):
            missing_packages.append(package_name)
    
    # Check if CatBoost is missing
    if not check_module("catboost"):
        catboost_required = True
    
    # Install missing packages
    if missing_packages or catboost_required:
        packages_to_install = missing_packages.copy()
        if catboost_required:
            print("\nCatBoost is required but not installed.")
        
        if packages_to_install:
            print(f"The following packages need to be installed: {', '.join(packages_to_install)}")
        
        # Ask for confirmation
        if input("\nDo you want to install these packages now? (y/n): ").lower() != 'y':
            print("Installation canceled. Note that the application may not work correctly without these packages.")
            return
        
        for package in packages_to_install:
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
        
        # Handle CatBoost installation separately
        if catboost_required:
            print("\n--- CatBoost Installation ---")
            
            # Check if Python version might be incompatible
            is_python_too_new = python_version.major == 3 and python_version.minor >= 12
            
            if is_python_too_new:
                print(f"⚠️ Warning: Python {python_ver_str} may not be supported by CatBoost yet.")
                print("CatBoost may not have pre-built wheels for this Python version.")
                
                if input("Would you like to try installing CatBoost anyway? (y/n): ").lower() != 'y':
                    print("Skipping CatBoost installation.")
                    print("The application will continue to work with the other ML models.")
                    return
            
            print("CatBoost requires Rust to be installed for some dependencies.")
            
            if check_rust_installed():
                print("✅ Rust is installed. Attempting to install CatBoost...")
                if install_package("catboost"):
                    print("✅ CatBoost installed successfully")
                else:
                    print("❌ Failed to install CatBoost with Rust available")
                    print("\nAlternative options:")
                    if is_python_too_new:
                        print(f"1. Try using Python 3.11 instead of {python_ver_str}")
                    print(f"2. The application will continue to work with other ML models")
            else:
                print("⚠️ Rust is not installed. Attempting to install pre-built CatBoost wheel...")
                if install_package("catboost", ["--only-binary", ":all:"]):
                    print("✅ CatBoost binary version installed successfully")
                else:
                    print("❌ Failed to install CatBoost binary version")
                    print("\nTo install CatBoost with all features, you need to:")
                    print("1. Install Rust from https://rustup.rs/")
                    print("2. Add Rust to your PATH environment variable")
                    print("3. Run this installer again or manually install with: pip install catboost")
                    
                    if is_python_too_new:
                        print(f"\nAlternatively, CatBoost may not support Python {python_ver_str} yet.")
                        print("Consider using Python 3.11 instead.")
                    
                    print("\nThe application will continue to work with the other ML models")
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