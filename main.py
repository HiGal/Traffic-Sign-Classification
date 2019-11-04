from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import time

if __name__ == '__main__':
    # with augmentation
    aug = True
    printed = False
    img_sizes = [-1, 4, 8, 16, 30, 32]
    accuracy = []
    perfomance = []
    print("Start reading dataset")
    train_X, train_y, test_X, test_y = train_test_split('./GTSRB/Final_Training/Images/')
    final_test_X, final_test_y = readTrafficSignsFinal('./GTSRB/Final_Test/')

    print("Dataset has been read")
    for size in img_sizes:
        if size == -1:
            aug = False
        else:
            aug = True
        if aug:
            print("\nWith Augmentation:")
            # Maybe that there is can be image with the same size
            show_padded_and_resized([train_X[0]], size=size)
            transformed_train_X = padding_and_resizing(train_X, size)
            transformed_test_X = padding_and_resizing(test_X, size)
            # data augmentation
            transformed_train_X, new_train_y = augmentation(transformed_train_X, train_y)
            new_final_test_X = normalize(padding_and_resizing(final_test_X, size))
            if not printed:
                show_train_freq(train_y, 'Before augmentation')
                show_train_freq(new_train_y, 'After augmentation')
                printed = True
        else:
            print("\nWithout Augmentation:")
            transformed_train_X = padding_and_resizing(train_X, 30)
            transformed_test_X = padding_and_resizing(test_X, 30)
            new_final_test_X = normalize(padding_and_resizing(final_test_X, 30))
            new_train_y = train_y
        new_train_X, new_test_X = normalize(transformed_train_X), normalize(transformed_test_X)
        if size == -1:
            print(f"Training the model on {30}x{30} image:")
        else:
            print(f"Training the model on {size}x{size} image:")

        start = time.time()
        rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rfc.fit(new_train_X, new_train_y)
        perfomance.append(time.time() - start)

        print(f"Validation score: {rfc.score(new_test_X, test_y)}")
        accuracy.append(rfc.score(new_final_test_X, final_test_y))
        print(f'Test score: {accuracy[-1]}')

        y_pred = rfc.predict(new_final_test_X)
        if size == 30:
            show_wrong_predictions(y_pred, final_test_y, new_final_test_X, size)
            aug = Blur(blur_limit=2, p=1)
            augment_and_show(aug, transformed_test_X[0], 'Blur')
            aug = RandomSunFlare(src_radius=20, p=1)
            augment_and_show(aug, transformed_test_X[3], 'Random Sun Flare')
            aug = ShiftScaleRotate(p=1)
            augment_and_show(aug, transformed_test_X[8], 'Shift Scale Rotate')
            aug = RandomBrightnessContrast(p=1)
            augment_and_show(aug, transformed_test_X[20], 'Random Sun Flare')
            aug = RandomRain(blur_value=2, drop_width=1, p=1)
            augment_and_show(aug, transformed_test_X[40], 'Random Rain')
            precision = precision_score(final_test_y, y_pred, average=None)
            recall = recall_score(final_test_y, y_pred, average=None)
            print(f'\nPrecision score: {precision}')
            plt.bar(list(range(43)), precision)
            plt.xlabel("Class IDs")
            plt.ylabel("Precision score")
            plt.show()
            print(f'\nRecall score: {recall}')
            plt.bar(list(range(43)), recall)
            plt.xlabel("Class IDs")
            plt.ylabel("Recall score")
            plt.show()
    plt.plot(img_sizes, accuracy, '.-')
    plt.ylabel("Accuracy")
    plt.xlabel("Image size")
    plt.show()
    plt.plot(img_sizes, perfomance, '.-')
    plt.ylabel("Training time (sec)")
    plt.xlabel("Image size")
    plt.show()
