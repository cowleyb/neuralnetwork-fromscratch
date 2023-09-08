import read_data
import pytorch_classifier
import matplotlib.pyplot as plt
import torch


def main():
  
  (x_train, y_train),(x_test, y_test) = read_data.load_data()
  plt.savefig("thing.png")

  model = pytorch_classifier.NeuralNetwork()
  pytorch_classifier.train(model, x_train, y_train)
  print("Done training")
  pytorch_classifier.test(model, x_test, y_test)
  print("Done testing")

  pytorch_classifier.show_predictions(model, x_test, y_test)
  print("Done showing predictions")
  # pytorch_classifier.show_mistakes(model, x_test, y_test)
  print("Done showing mistakes")


if __name__ == '__main__':
  main()
  

