import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Optional


# Enhanced training loop with early stopping and gradient clipping
def train_model(model, train_loader, val_loader, num_epochs=300, device='cuda', 
                patience=15, clip_grad=1.0):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for bags, lens, labels in train_loader:
            bags = bags.to(device)
            lens = lens.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(bags, lens)
            loss = criterion(logits, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            train_loss += loss.item()
            total += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for bags, lens, labels in val_loader:
                bags, lens, labels = [t.to(device) for t in (bags, lens, labels)]
                logits = model(bags, lens)
                
                val_loss += criterion(logits, labels).item()
                val_total += labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'best_model.pth')
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve == patience:
        #         print(f'\nEarly stopping at epoch {epoch + 1}')
        #         break
        
        print(f'Epoch {epoch + 1:03d}/{num_epochs}:'
              f'  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%'
              f'  Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
    
    print(f'Training complete. Best validation accuracy: {best_val_acc:.2f}%')
