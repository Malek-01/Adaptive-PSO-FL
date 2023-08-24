from plotting import plot
from benchFunctions import functions_dict as fd

import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Liste des noms de fonctions
function_names = list(fd.keys())

def on_select(event):
    selected_function = combobox.get()
    root.update()  # Mettre à jour l'interface pour afficher la barre de progression

    # Exécuter la fonction dans un thread pour ne pas bloquer l'interface
    t = threading.Thread(target=run_function, args=(selected_function,))
    t.start()

def run_function(selected_function):
    # Créer une barre d'avancement
    progress_bar = ttk.Progressbar(root, orient='horizontal', length=300, mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()  # Démarrer la barre d'avancement

    root.update()

    fac = plot(fd[selected_function])
    dialog = tk.Toplevel(root)
    dialog.title("Tableau des Résultats")
    table_text = f"""
                {"    ":<10s}{"min":^30s}|{"mean":^30s}|{"std deviation":^30s}
           ------------------------------------------------------------------------------------------------\n
                {"CPSO":<10s}|{fac[0][0]:^30,.30f}|{fac[0][1]:^30,.30f}|{fac[0][2]:^30,.30f}\n
                {"SPSO":<10s}|{fac[1][0]:^30,.30f}|{fac[1][1]:^30,.30f}|{fac[1][2]:^30,.30f}\n
                {"HCPSO":<10s}|{fac[2][0]:^30,.30f}|{fac[2][1]:^30,.30f}|{fac[2][2]:^30,.30f}\n
                {"EPSO":<10s}|{fac[3][0]:^30,.30f}|{fac[3][1]:^30,.30f}|{fac[3][2]:^30,.30f}\n
    """
    label = tk.Label(dialog, text=table_text, padx=10, pady=10)
    label.pack()
    
    progress_bar.stop()  # Arrêter la barre d'avancement
    progress_bar.destroy()

    print(table_text)

root = tk.Tk()
root.title("Function Selector")
root.geometry("600x450")

label = tk.Label(root, text="Choose a function:")
label.pack(pady=10)

combobox = ttk.Combobox(root, values=function_names)
combobox.pack(pady=5)
combobox.bind("<<ComboboxSelected>>", on_select)

root.mainloop()