
def own_resnet_model():
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          # Prep layer
          self.preplayer = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1,stride=1, bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU()
            
          ) 

        
          # LAYER 1
          self.layer1 = nn.Sequential(
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1,bias=False),
              nn.MaxPool2d(2, 2),
              nn.BatchNorm2d(128),
              nn.ReLU()
            
          )

          # Resnet layer 1 Block
          self.R1 = nn.Sequential(
              nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1,bias=False),
              nn.BatchNorm2d(128),
              nn.ReLU(),

              nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1,bias=False),
              nn.BatchNorm2d(128),
              nn.ReLU()
            
          )


          # LAYER 2
          self.layer2 = nn.Sequential(
              nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1,bias=False),
              nn.MaxPool2d(2, 2),
              nn.BatchNorm2d(256),
              nn.ReLU()
            
          )


          # LAYER 3
          self.layer3 = nn.Sequential(
              nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,bias=False),
              nn.MaxPool2d(2, 2),
              nn.BatchNorm2d(512),
              nn.ReLU()
            
          )


            # Resnet layer 3 Block
          self.R2 = nn.Sequential(
              nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,bias=False),
              nn.BatchNorm2d(512),
              nn.ReLU(),

              nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,bias=False),
              nn.BatchNorm2d(512),
              nn.ReLU()
            
          )

          # Maxpool with 4*4 kernel
          self.pool4 = nn.MaxPool2d(4, 4)

                  
          #Fully connected layer
          self.fc1 = nn.Sequential( nn.Linear(1024, 10))


        
      def forward(self, x):

          x  = self.preplayer(x) 
          # print('printing preplayer',x.shape)

          x  = self.layer1(x) 
          # print('printing layer 1',x.shape)

          r1 = self.R1(x)
          # print('Printing r1',r1.shape)

          x  = self.layer2(torch.cat([x,r1],dim=1))
          # print('printing layer 2',x.shape)

          x  = self.layer3(x)
          # print('printing layer 3',x.shape)

          r2 = self.R2(x)
          # print('printing r2',r2.shape)

          x  = self.pool4(torch.cat([x,r2],dim=1))
          # print('printing after maxpool',x.shape)

          x = x.view(-1, 1024*1*1)

          x = self.fc1(x)
          # print('printing fc',x.shape)
          
          return F.log_softmax(x, dim=-1)
  print("Model was build as per given architucture....")
  return Net