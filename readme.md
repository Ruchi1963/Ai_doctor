create virture envioremnt 
install ffmeg : mannualy download it and add bin file to path or conda install -c conda-forge ffmpeg
on cmd conda activate myenv(enviorment name )

starting m code ko run krne m ek do baar image upload krne k nbaad screen freeze ho ja rha jb file picker hota kyonki gradio use kr rha tha filepath as a type jiske chlte gradio disk m image ko temporary store krta tha aur reuse krta use phr clear nhi krta tha ....jb gradio ko delete kr k reinstall hone p ek do baar kaam krta phr same issue then  hm type pil ko use kiya instead of filepath jis m gradio memory m image store krta hai dependency kam krne k liye
grok use kr k public url nhi create ho rha tha issue aa rha tha har time maybe uske server k chlte toh ngrok s create kiye.... ngork ko install kr k uska aut token s initialize krne k baad kaam krne lag gya

steps:
create a virture enviorment using conda like "conda create --name myenv python=3.10"
activate virture enviorment : "conda activate myenv
install dependencies : "pip install -r requirement.txt"
ngrok config add-authtoken 2w3HonwOC6UqmGWYcQ4hVyAkJBB_41pBzg3uMyxdrkaf8ET9N
run main file: "python gradio_app.py"
#Run all on terminal
#Replace api with your own api in env file
