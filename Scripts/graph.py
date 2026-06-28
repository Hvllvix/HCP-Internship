import subprocess
import os

def build_git_graph(commit_limit=15):
    """
    Fetches the local git history and formats it into a Graphviz DOT string
    with a minimalist, zinc-themed aesthetic.
    """
    try:
        cmd = f'git log --graph --pretty=format:"%h|%p|%s" -n {commit_limit}'
        log_data = subprocess.check_output(cmd, shell=True).decode("utf-8")
        
        dot_lines = [
            "digraph GitGraph {",
            "    rankdir=LR;",
            "    node [shape=box, style=filled, fillcolor=\"#f4f4f5\", fontname=\"sans-serif\", fontsize=10, color=\"#e4e4e7\"];",
            "    edge [color=\"#a1a1aa\", penwidth=1.5];",
            "    bgcolor=\"transparent\";"
        ]
        
        for line in log_data.split('\n'):
            line = line.strip()
            if not line or '*' not in line:
                continue
                
            clean_line = line.replace('*', '').strip()
            parts = clean_line.split('|')
            
            if len(parts) == 3:
                hash_id, parents, message = parts
                label = message[:25].replace('"', '\\"') 
                dot_lines.append(f'    "{hash_id}" [label="{label}"];')
                
                if parents:
                    for parent_id in parents.split():
                        dot_lines.append(f'    "{parent_id}" -> "{hash_id}";')
        
        dot_lines.append("}")
        return "\n".join(dot_lines)
    
    except Exception as error:
        return f'digraph Error {{ node [shape=box]; Error [label="Git Error: {str(error)}"]; }}'

def save_graph_to_file():
    """
    Executes the graph build and saves the DOT syntax to Others/git_graph.txt
    """
    graph_content = build_git_graph()
    output_path = os.path.join("Others", "git_graph.txt")
    
    # Ensure the Others directory exists
    os.makedirs("Others", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(graph_content)
        
    print(f"Success: Git graph DOT syntax saved to {output_path}")

if __name__ == "__main__":
    save_graph_to_file()