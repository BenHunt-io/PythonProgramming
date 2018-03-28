
videos = []

with open("data.txt", 'r') as f:
      for path in f.readlines():
        path = path.strip()
        print(path + str(type(path)))
        if path:
          videos.append(path.strip()) #

print(videos)