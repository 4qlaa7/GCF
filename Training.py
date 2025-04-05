import torch

def train(GCF_model,device,train_loader,val_loader,num_epochs,criterion,optimizer):
  for epoch in range(num_epochs):
    # Training
    GCF_model.train()
    total_correct_train = 0
    total_samples_train = 0
    total_loss_train = 0

    for images, labels in train_loader:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        emotion_output = GCF_model(images)

        # Calculate loss
        loss = criterion(emotion_output, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy and loss
        _, predicted_train = torch.max(emotion_output, 1)
        total_correct_train += (predicted_train == labels).sum().item()
        total_samples_train += labels.size(0)
        total_loss_train += loss.item()

    # Calculate training accuracy and loss after each epoch
    accuracy_train = "%.2f" % (100 * (total_correct_train / total_samples_train))
    average_loss_train = "%.2f" % (total_loss_train / len(train_loader))

    # Validation
    GCF_model.eval()  # Set the model to evaluation mode
    total_correct_val = 0
    total_samples_val = 0
    total_loss_val = 0

    with torch.no_grad():  # Disable gradient computation during validation
        for images_val, labels_val in val_loader:
            # Move data to GPU
            images_val, labels_val = images_val.to(device), labels_val.to(device)

            emotion_output_val = GCF_model(images_val)
            loss_val = criterion(emotion_output_val, labels_val)
            _, predicted_val = torch.max(emotion_output_val, 1)
            total_correct_val += (predicted_val == labels_val).sum().item()
            total_samples_val += labels_val.size(0)
            total_loss_val += loss_val.item()

    # Calculate validation accuracy and loss
    accuracy_val = "%.2f" % (100 *(total_correct_val / total_samples_val))
    average_loss_val = "%.2f" % (total_loss_val / len(val_loader))
    # Print training and validation statistics after each epoch
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {average_loss_train}, Train Acc: {accuracy_train}%, '
          f'Val Loss: {average_loss_val}, Val Acc: {accuracy_val}%')

