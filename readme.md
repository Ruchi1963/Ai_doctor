create virture envioremnt 
install ffmeg : mannualy download it and add bin file to path or conda install -c conda-forge ffmpeg
on cmd conda activate myenv(enviorment name )

starting m code ko run krne m ek do baar image upload krne k nbaad screen freeze ho ja rha jb file picker hota kyonki gradio use kr rha tha filepath as a type jiske chlte gradio disk m image ko temporary store krta tha aur reuse krta use phr clear nhi krta tha ....jb gradio ko delete kr k reinstall hone p ek do baar kaam krta phr same issue then  hm type pil ko use kiya instead of filepath jis m gradio memory m image store krta hai dependency kam krne k liye

steps:
create a virture enviorment using conda like "conda create --name myenv python=3.10"
activate virture enviorment : "conda activate myennv
install dependencies : "pip install -r requirement.txt"
run maain file: "python gradio_app.py"
