urls = open("speclist_urlext.txt", "r")
speclist = open("speclist.txt","w")
for url in urls:
    speclist.write(url[5:])
