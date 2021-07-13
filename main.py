from model_inversion.get_image import get_image
# from model_inversion.main import get_target_model_accuracy

from PIL import Image
from membership.main import train_shallow_and_attack_model
from membership.validate import test_membership_model

if __name__ == '__main__':
    print('Enter 1 for model inversion ')
    print('Enter 2 for membership inference ')

    x = input()
    x = int(x)
    if x == 1:
        print('Enter a number between 1 and 40 ')
        x = input()
        if 1 <= int(x) <= 40:
            get_image(int(x) - 1)
            a = Image.open("./final_image.png")
            a.show()
    elif x == 2:
        print("Enter 1 for training shadow model otherwise enter any other number")
        x = input()
        if x == 1:
            print('training attack and shallow model on mnist dataset ')
            train_shallow_and_attack_model()
        else:
            print("skipped training for shadow model")
        print('testing membership model')
        test_membership_model()
