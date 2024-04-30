import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from PIL import Image, ImageTk
import threading
from classifier import full_system

def show_frame(frame):
    frame.tkraise()

def update_image_display(image_path):
    try:
        img = Image.open(image_path)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        panel_image.config(image=img)
        panel_image.image = img
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")

def classify_image(filepath):
    image = cv2.imread(filepath)
    classification = full_system(image)
    label_result.config(text=f"Classification: {classification}", fg="green", font=('Helvetica', 12, 'bold'))

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("JPEG Images", "*.jpeg")])
    if not filepath:
        return
    show_frame(frame_single)
    threading.Thread(target=lambda: update_image_display(filepath)).start()
    threading.Thread(target=lambda: classify_image(filepath)).start()

def process_file(directory_path, file):
    try:
        image = cv2.imread(os.path.join(directory_path, file))
        classification = full_system(image)
        return (file, classification)
    except Exception as e:
        return (file, "Error")

def open_directory():
    directory_path = filedialog.askdirectory()
    if not directory_path:
        return
    files = [f for f in os.listdir(directory_path) if f.lower().endswith('.jpeg')]
    if not files:
        messagebox.showinfo("Info", "No JPEG images found in the selected directory.")
        return
    show_frame(frame_directory)
    progress = ttk.Progressbar(frame_directory, orient='horizontal', mode='determinate', maximum=len(files))
    progress.pack(pady=10)
    tree.delete(*tree.get_children())
    for file in files:
        tree.insert("", "end", values=process_file(directory_path, file))
        progress.step()
        root.update()
    progress.destroy()



root = tk.Tk()
root.title("Bottle Bottom Classifier")
root.configure(bg='white')

style = ttk.Style()
style.configure("TButton", font=('Helvetica', 10), padding=6)
style.configure("TLabel", background='white', font=('Helvetica', 10))
style.configure("Treeview", highlightthickness=0, bd=0, font=('Helvetica', 10))  # Treeview style
style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))  # Treeview heading style
style.map("TButton", foreground=[('pressed', 'red'), ('active', 'blue')], background=[('pressed', '!disabled', 'black'), ('active', 'white')])

# Main Menu Frame
frame_main = tk.Frame(root, bg='white')
frame_main.grid(row=0, column=0, sticky='news')

btn_single = ttk.Button(frame_main, text="Classify Single Image", command=lambda: open_file())
btn_single.pack(pady=10)

btn_directory = ttk.Button(frame_main, text="Classify Directory", command=lambda: open_directory())
btn_directory.pack(pady=10)

# Single Image Frame
frame_single = tk.Frame(root, bg='white')
frame_single.grid(row=0, column=0, sticky='news')

btn_back_single = ttk.Button(frame_single, text="Back to Main Menu", command=lambda: show_frame(frame_main))
btn_back_single.pack()

label_result = tk.Label(frame_single, text="Classification: None", bg='white')
label_result.pack()

panel_image = tk.Label(frame_single, bg='white')
panel_image.pack()

# Directory Frame
frame_directory = tk.Frame(root, bg='white')
frame_directory.grid(row=0, column=0, sticky='news')

btn_back_directory = ttk.Button(frame_directory, text="Back to Main Menu", command=lambda: show_frame(frame_main))
btn_back_directory.pack()

tree_scroll = tk.Scrollbar(frame_directory)
tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

tree = ttk.Treeview(frame_directory, columns=('File Name', 'Classification'), show='headings', yscrollcommand=tree_scroll.set)
tree.heading('File Name', text='File Name')
tree.heading('Classification', text='Classification')
tree.pack(fill='both', expand=True)

tree_scroll.config(command=tree.yview)

# Raise the main menu frame initially
show_frame(frame_main)

root.mainloop()