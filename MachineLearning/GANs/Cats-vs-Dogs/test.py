"""import csv

with open('losses.csv', 'a+', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # Generator Summary
    writer.writerow(['Generator'])
    writer.writerow(['Dense', 'Neurons: 18*18*256', 'input: (1000,)', 'batch_normlization', 'activation: leaky_relu'])
    writer.writerow(['Conv2DTranspose', 'Neurons: 128', '(5, 5)', 'strides: (1, 1)', 'batch_normlization', 'activation: leaky_relu'])
    writer.writerow(['Conv2DTranspose', 'Neurons: 64', '(5, 5)', 'strides: (1, 1)', 'batch_normlization', 'activation: leaky_relu'])
    writer.writerow(['Conv2DTranspose', 'Neurons: 3', '(5, 5)', 'strides: (1, 1)', 'batch_normlization', 'activation: leaky_relu'])

    writer.writerow([])

    writer.writerow([])

    # Discriminator Summary
    writer.writerow(['Discriminator'])
    writer.writerow(['Conv2D', 'Neurons: 128', '(5, 5)', 'strides: (2, 2)', 'padding: same', 'input: (72, 72, 3)', 'activation: leaky_relu', 'Dropout: 0.3'])
    writer.writerow(['Conv2D', 'Neurons: 128', '(5, 5)', 'strides: (2, 2)', 'padding: same', 'activation: leaky_relu', 'Dropout: 0.3'])

    writer.writerow([])

    writer.writerow([])"""

for i in range(0, 100000):
    print(i)