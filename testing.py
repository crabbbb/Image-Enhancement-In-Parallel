# Open a file in write mode. If the file doesn't exist, it will be created.
with open("testing/hello_world.txt", "w") as file:
    # Write "Hello, World!" to the file
    file.write("Hello, World! Testing write file, you should be able to read this message in the ipykernel.")

print("File write done.")