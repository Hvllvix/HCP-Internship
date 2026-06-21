import os

def print_directory_tree(dir_path, prefix=""):
    try:
        # Get all items, sorted for clean output
        items = sorted(os.listdir(dir_path))
    except PermissionError:
        # Skip directories where permission is denied
        return

    # Exclude the .git folder from the list of items
    items = [item for item in items if item != '.git']

    for index, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = (index == len(items) - 1)
        
        # Use standard branch connectors
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{item}")
        
        # If the item is a directory, recursively print its contents
        if os.path.isdir(path):
            extension_prefix = "    " if is_last else "│   "
            print_directory_tree(path, prefix + extension_prefix)

if __name__ == '__main__':
    print(".")
    print_directory_tree('.')