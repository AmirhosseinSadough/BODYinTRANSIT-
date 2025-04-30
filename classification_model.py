import torch
import torch.nn as nn 

from decorrelation.decorrelation import DecorConv2d

        
# Self-attention block
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch, d_model, time]
        x = x.permute(2, 0, 1)  # [time, batch, d_model]
        attn_output, _ = self.mha(x, x, x)
        x = self.norm(x + attn_output)
        ffn_output = self.ffn(x.transpose(0, 1)).transpose(0, 1)
        x = self.norm2(x + ffn_output)
        return x.permute(1, 2, 0)  # [batch, d_model, time]

# Cross-attention block
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, xc):
        # x, xc: [batch, d_model, time]
        x = x.permute(2, 0, 1)    # [time, batch, d_model]
        xc = xc.permute(2, 0, 1)  # [time, batch, d_model]
        attn_output, _ = self.mha(query=x, key=xc, value=xc)
        x = self.norm(x + attn_output)
        return x.permute(1, 2, 0)  # [batch, d_model, time]

class InputProjection(nn.Module):
    def __init__(self, num_sensors, d_model):
        super().__init__()
        self.proj = nn.Conv1d(num_sensors, d_model, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        return x
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, xc):
        # x, xc: [batch, d_model, time]
        x = x.permute(2, 0, 1)    # [time, batch, d_model]
        xc = xc.permute(2, 0, 1)  # [time, batch, d_model]
        attn_output, _ = self.mha(query=x, key=xc, value=xc)
        x = self.norm(x + attn_output)
        return x.permute(1, 2, 0)  # [batch, d_model, time]
    
class Classifier(nn.Module):
    def __init__(self, d_model, num_classes=3):  # Changed rnn_hidden_size to d_model
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),            
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, d_model] (after averaging over time)
        return self.fc(x)  # [batch, num_classes]

class mahcross_n(nn.Module):
    def __init__(self, num_sensors, d_model=512, num_heads=8, num_layers=2, num_classes=3, dropout_prob=0.0, rnn_hidden_size=256):
        super(mahcross_n, self).__init__()
        # Input projection
        self.num_sensors = num_sensors  # Add this line
        self.dec = DecorConv2d(1, 1, kernel_size=(3, 3), stride=1, padding=0, decor_lr=0.01, kappa=1e-3, method= 'standard')
        self.dec1 = DecorConv2d(1, 1, kernel_size=(3, 3), stride=1, padding=0, decor_lr=0.01, kappa=1e-3, method= 'standard')
        self.proj_x = InputProjection(num_sensors, d_model)
        self.proj_xc = InputProjection(num_sensors, d_model)
        
        # Self-attention blocks
        self.x_self_attn = nn.ModuleList([SelfAttentionBlock(d_model, num_heads, dropout_prob) for _ in range(num_layers)])
        self.xc_self_attn = nn.ModuleList([SelfAttentionBlock(d_model, num_heads, dropout_prob) for _ in range(num_layers)])
        
        # Cross-attention block (still defined but will be skipped in forward)
        self.cross_attn = CrossAttentionBlock(d_model, num_heads, dropout_prob)
        
        # RNN layer (GRU)
        self.rnn = nn.GRU(input_size=d_model, hidden_size=rnn_hidden_size, num_layers=2, batch_first=True, dropout=dropout_prob)
        
        # Classifier
        self.classifier = Classifier(rnn_hidden_size, num_classes)

    def forward(self, x, xc):
        x = x.squeeze(1).permute(0, 2, 1)
        xc = xc.squeeze(1).permute(0, 2, 1)

        x = self.proj_x(x)
        xc = self.proj_xc(xc)

        for layer in self.x_self_attn:
            x = layer(x)
        for layer in self.xc_self_attn:
            xc = layer(xc)

        x = self.cross_attn(x, xc)  

        x = x.permute(0, 2, 1)
        # print("pre RNN", x.shape)
        rnn_output, hidden = self.rnn(x)
        x = hidden[-1]
        # print("post RNN", x.shape)

        return self.classifier(x)    
    


if __name__ == "__main__":

    x = torch.randn(32, 1, 300, 69)  
    xc = torch.randn(32, 1, 300, 69)
    model = mahcross_n(num_sensors=69, d_model=256, num_heads=8, num_layers=2, num_classes=3, rnn_hidden_size=128, dropout_prob=0.1)

    output = model(x, xc)
    print("Output shape:", output.shape)  # Should be [32, 3] for 3 classes
    


# model = mahcross_n(num_sensors=69, d_model=256, num_heads=8, num_layers=2, num_classes=3, rnn_hidden_size=128, dropout_prob=0.1)

# # Load checkpoint
# checkpoint = torch.load('checkpoint_f1_0.4406_epoch_66.pth', map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])  # or checkpoint['state_dict'] depending on how it's saved

# # Print architecture
# # print(model)