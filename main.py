from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
from google_trans_new import google_translator
import os
import csv
import random
SEED = 34
tf.random.set_seed(SEED)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

def translate(items):
    translator = google_translator()
    if type(items) == "list":
        ret = []        
        for item in items:
            ret.append(translator.translate(item, lang_tgt = 'hu'))
        return ret
    else:
        items.replace("<|endoftext|>", "")
        return translator.translate(items, lang_tgt = 'hu')
def selectRandom (items, minm, maxm):
    count = random.randint(minm, maxm)
    return random.sample(items, count)

def addImages(txt, imgs):
    try:
        ll = txt.split("\n")
        img = random.choice(imgs)
        img2 = random.choice(imgs)
        cnt1 = (len(ll) // 2) //2
        cnt2 = cnt1 + (len(ll) // 2)    
        out = "\n".join(ll[0:cnt1])
        out = out + " <img src=" + img + ">"
        out = out + "\n".join(ll[cnt1:cnt2])
        out = out + " <img src=" + img2 + ">"
        out = out + "\n".join(ll[cnt2:])
        return out
    except Exception as e:
        print(e)
        return txt 

def highlight_Article(art, high):
    for h in high:
        if len (h) > 3:
            fin = "<b>" + h + "</b>"
            art = art.replace(h, fin)
    return art

#maximum number of words in output text
MAX_LEN = 500
st_head = ["<h1>", "<h2>", "<h3>"]
en_head = ["</h1>", "</h2>", "</h3>"]
try:
    os.remove("output.csv")
except:
    pass
outpt = csv.writer(open('output.csv', 'w',  encoding='utf-8'))
outpt.writerow(["keyword", "GUID", "Description", "Tags", "Article","Article-english", "Category"])


# open text file
with open('u\\tx.txt') as f0:#open('tx654.txt') as f0:#open('u\\te.txt') as f0:#open('tx654.txt') as f0:
    txt = f0.readlines()

# open title file
with open('u\\ti.txt') as f1:#open('ttt165.txt') as f1:#open('u\\ti.txt') as f1: #open('ttt165.txt') as f1:
    titles = f1.readlines()

# open keywords file
with open('u\\k.txt') as f2:# open('kk654.txt') as f2:#open('u\\k.txt') as f2: #open('kk654.txt') as f2:
    keywords = f2.readlines()

# open images file
with open('u\\i.txt') as f3:# open('im95.txt') as f3:#open('u\\i.txt') as f3: #open('im95.txt') as f3:
    images = f3.readlines()

for xm, (title,tt) in enumerate (zip(titles,txt)): 
    keyword = translate(keywords[xm % len(keywords)]) 
    print("=" * 20) 
    tt = tt[0:tt.rindex(".")]
    usd_titles = []
    #tt= tt.replace("\n","")
    title = title.replace("\n","") 
    usd_titles.append(title)
    title = translate(title)
    highlight = title.split(" ")
    highlight.extend(keyword.split(" "))

    


    print("Generating text for: ", title)
    print("Input Sentence: ", tt)               
    print("=" * 20)
    inps = tt.split(".")
    
    imgs = random.sample(images, min(len(inps)-1,len(images)))
    tits = random.sample(titles, min(len(inps)-1,len(titles)))
    tmp2 = random.sample(keywords, min(len(inps)-1,len(keywords)))
    kkw = [translate(k.replace("\n",'')) for k in tmp2]

    temp = [translate(t.replace("\n","")).split(" ") for t in tits]
    [highlight.extend(tt) for tt in temp]



    article = ""
    art_eng = ""
    for enm,inp in enumerate(inps):
        
        while True:
            input_ids = tokenizer.encode(inp, return_tensors='tf')
            sample_outputs = GPT2.generate(
                                        input_ids,
                                        do_sample = True, 
                                        max_length = 2*MAX_LEN,                              #to test how long we can generate and it be coherent
                                        #temperature = .7,
                                        top_k = 50, 
                                        top_p = 0.85, 
                                        num_return_sequences = 5
            )
            if not "<|endoftext|>" in tokenizer.decode(sample_outputs[0], skip_special_tokens = True):
                break                
        amb = inp + tokenizer.decode(sample_outputs[0], skip_special_tokens = True)
        amb = amb[0:amb.rindex(".")] + "."
        
        art_eng += inp + amb
        article += highlight_Article(translate(inp + amb),highlight)
        if enm < len(inps)-1:                    
            img = imgs[enm].replace("\n","")
            article += "\n <img src=" + img + " alt = " + keyword + "> \n"
            art_eng += "\n <img src=" + img + " alt = " + keyword + "> \n"                    
            
            t2 = tits[enm].replace("\n","")
                            
            hd = random.randint(0,2)
            if sameKeyword: 
                kk = keyword
            else:
                kk = kkw[enm]
            article += st_head[hd] + kk + " - " + translate(t2) + en_head[hd] + "\n" 
            
        
    title = keyword +" - "+ title
    print(art_eng)          
    #article = article.replace(" <| Endoftext |>", "")  #
    #article = article.replace("<|endoftext|>", "")
    #article = translate(article)
    #article = highlight_Article(article,highlight)
    tags = translate(",".join(selectRandom(keywords,3,4)))
    categories = translate(",".join(selectRandom(keywords,1,2)))
    #article = addImages(article,images)
    outpt.writerow([keyword, xm+1, title, tags, article,art_eng, categories])




'''
input_sequence = "I don't know about you, but there's only one thing I want to do after a long day of work"

# encode context the generation is conditioned on
input_ids = tokenizer.encode(input_sequence, return_tensors='tf')

sample_outputs = GPT2.generate(
                              input_ids,
                              do_sample = True, 
                              max_length = 2*MAX_LEN,                              #to test how long we can generate and it be coherent
                              #temperature = .7,
                              top_k = 50, 
                              top_p = 0.85, 
                              num_return_sequences = 5
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_outputs[0])) #, skip_special_tokens = True
'''