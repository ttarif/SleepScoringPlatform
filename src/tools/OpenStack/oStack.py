import os
import openstack
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

def connect_to_cloud(cloud_name='openstack'):
    # Initialize the connection
    return openstack.connect(cloud_name=cloud_name)

def get_objects_in_container(conn, container_name):
    # Get the list of objects in the container
    objects = conn.object_store.objects(container_name)
    for obj in objects:
        if obj.name.startswith('jkb/')and (obj.name.endswith('.edf') or obj.name.endswith('.csv')):
            yield obj

def download_object(conn, obj, directory):
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Download the object's data
    data = conn.object_store.download_object(obj)

    # Check if the data is None
    if data is None:
        print(f"No data found for object {obj.name}")
        return

    # Create the subdirectories in the path if they don't exist
    os.makedirs(os.path.join(directory, os.path.dirname(obj.name)), exist_ok=True)

    # Save the data to a file in the specified directory
    with open(os.path.join(directory, obj.name), 'wb') as f:
        f.write(data)

def main():
    conn = connect_to_cloud()
    container_name = 'data'  # replace with your container name
    objects = get_objects_in_container(conn, container_name)

    # Create a dictionary to store the objects grouped by their names
    object_dict = defaultdict(list)

    # Group the objects by their names
    for obj in objects:
        parts = obj.name.split('/')
        if len(parts) > 1:
            name_parts = parts[1].split('_')
            if len(name_parts) > 1:
                name = name_parts[1]
                object_dict[name].append(obj)

    # Create a Tkinter window
    window = tk.Tk()
    window.title("OpenStack File Downloader")
    window.geometry("800x600")  # Set the default size of the window

    # Create a listbox to display the unique names
    name_listbox = tk.Listbox(window, selectmode=tk.SINGLE)
    for name in object_dict.keys():
        name_listbox.insert(tk.END, name)
    name_listbox.pack()

    # Create a canvas and a scrollbar
    canvas = tk.Canvas(window)
    scrollbar = tk.Scrollbar(window, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a frame to hold the checkboxes and put it in the canvas
    checkbox_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=checkbox_frame, anchor='nw')

    # Update the checkbox frame when a name is selected
    def on_name_select(event):
        # Clear the checkbox frame
        for widget in checkbox_frame.winfo_children():
            widget.destroy()

        selected_name = name_listbox.get(name_listbox.curselection())
        for obj in object_dict[selected_name]:
            # Create a checkbox for each file
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(checkbox_frame, text=obj.name.split('/')[1], variable=var)
            checkbox.var = var
            checkbox.pack(fill=tk.X)

        # Update the scroll region of the canvas
        checkbox_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox('all'))

    name_listbox.bind('<<ListboxSelect>>', on_name_select)

    # Pack the canvas and the scrollbar
    canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
    scrollbar.pack(fill=tk.Y, side=tk.RIGHT)

    # Create a progress bar
    progress = ttk.Progressbar(window, length=200, mode='determinate')
    progress.pack()

    # Download the selected files when a button is clicked
    def on_download_click():
        selected_name = name_listbox.get(name_listbox.curselection())
        selected_files = [obj for checkbox, obj in zip(checkbox_frame.winfo_children(), object_dict[selected_name]) if checkbox.var.get()]
        directory = filedialog.askdirectory()  # ask the user to select a directory

        # Initialize the progress bar
        progress['maximum'] = len(selected_files)
        progress['value'] = 0

        for obj in selected_files:
            download_object(conn, obj, directory)
            progress['value'] += 1  # Update the progress bar

        messagebox.showinfo("Download complete", "The selected files have been downloaded.")

    download_button = tk.Button(window, text="Download selected files", command=on_download_click)
    download_button.pack()

    # Start the Tkinter event loop
    window.mainloop()

if __name__ == '__main__':
    main()
