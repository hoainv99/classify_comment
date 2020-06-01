from CommentSemantic import CommentSemantic


def main():
  model = CommentSemantic()
  while True:
    comment = input('Type your test: ok ')
    if comment == '':
      break
    label = int(model.predict(comment))
    if label == 0:
      print("Good type !")
    else: print("Bad type !")
  print("Program stopped !")


if __name__ == '__main__':
  main()