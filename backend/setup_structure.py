import os

# Estructura de carpetas a crear
folders = [
    "app",
    "app/core",
    "app/agents",
    "app/services",
    "app/models"
]

# Archivos vacíos a crear (para que Python reconozca los módulos)
files = [
    "app/__init__.py",
    "app/main.py",
    "app/core/__init__.py",
    "app/core/config.py",
    "app/agents/__init__.py",
    "app/agents/interviewer.py",
    "app/agents/investigator.py",
    "app/agents/architect.py",
    "app/agents/builder.py",
    "app/agents/writer.py",
    "app/services/__init__.py",
    "app/services/chroma_service.py",
    "app/models/__init__.py"
]

def create_structure():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"🚀 Creando estructura en: {base_dir}")

    # Crear carpetas
    for folder in folders:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"✅ Carpeta creada: {folder}")
        else:
            print(f"ℹ️  Carpeta ya existe: {folder}")

    # Crear archivos
    for file in files:
        path = os.path.join(base_dir, file)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                pass # Crear archivo vacío
            print(f"✅ Archivo creado: {file}")
        else:
            print(f"ℹ️  Archivo ya existe: {file}")

    print("\n🏁 Estructura lista para la refactorización.")

if __name__ == "__main__":
    create_structure()