#!pip install gym
#!pip install torchsummary
#!pip install gym pyvirtualdisplay > /dev/null 2>&1
#!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
import gym 
import torch
import numpy
import warnings 
import torchvision
import torch.nn.functional as F
'''
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

import math
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env
'''
from torch import nn,optim
#from torchsummary import summary
from collections import deque
from torchvision import datasets,transforms
from torch.distributions import Categorical
from torch.nn.functional import one_hot,log_softmax, softmax,normalize
warnings.filterwarnings("ignore")
class HyperParameters:
  BATCH_SIZE=32
  LR=5e-3
  GAMMA=0.999
  RENDER=False
  EPOCHS = 5000
  BETA = 0.1
  
class Agent(nn.Module):
  def __init__(self, action_space:int):
      super(Agent,self).__init__()
      x=8
      self.ln1 = nn.Linear(in_features=80*80*3,out_features = 100)
      self.ln2 = nn.Linear(in_features=100,out_features=50)
      self.ln3 = nn.Linear(in_features=50,out_features=action_space)
      self.flatten = nn.Flatten()
  def forward(self,x):
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.tensor([0.25],device=device)
    x=self.flatten(x)
    x=F.prelu(self.ln1(x),weight)
    x=F.prelu(self.ln2(x),weight)
    x=F.prelu(self.ln3(x),weight)
    return x

'''
class Agent(nn.Module):
  def __init__(self, action_space:int):
      super(Agent,self).__init__()
      #Input is 80x80 image 
      self.conv1 = nn.Conv2d(in_channels=3,out_channels=2,kernel_size=1,stride=1,padding=0,padding_mode='zeros')
      self.bn1 = nn.BatchNorm2d(num_features=2)
      #Input is 80x80x3
      self.conv2 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=4,stride=2,padding=1,padding_mode='zeros')
      self.bn2 = nn.BatchNorm2d(num_features=1)
      #Input is 40x40x1
      self.flatten = nn.Flatten()
      self.ln1 = nn.Linear(in_features=40*40,out_features = action_space)
      self.prelu = nn.PReLU()
      self.prelu1 =  nn.PReLU()
      self.prelu2 =  nn.PReLU()
  def forward(self,x):
    x=self.prelu1(self.bn1(self.conv1(x)))
    x=self.prelu2(self.bn2(self.conv2(x)))
    x=self.flatten(x)
    x=self.prelu(self.ln1(x))
    return x
'''
class PG_RL:
  def __init__(self):
    self.epochs = HyperParameters.EPOCHS 
    self.batch_size = HyperParameters.BATCH_SIZE
    self.lr = HyperParameters.LR
    self.gamma = HyperParameters.GAMMA
    self.beta =  HyperParameters.BETA
    self.render = HyperParameters.RENDER
    #self.env = wrap_env(gym.make('Pong-v0'))
    self.env = gym.make('Pong-v0')
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Instantiating the agent with the output is equivalent to the action space 
    self.agent= Agent(action_space=self.env.action_space.n).to(self.device)
    self.optimiser = optim.Adam(params=self.agent.parameters(), lr=self.lr)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser,mode='min',factor=0.1,patience=10,verbose=True)
    self.total_reward = deque([],maxlen=200)

  def PreProcessing(self,image):

    image = image[35:195] #Cropping the image 
    image = image[::2,::2,:] #Downsampling the image
    #image[image == 144] = 0 # Erasing Background value 1
    #image[image == 109] =0 # Erasing Background value 2
    #image[image !=0] = 1 #Everything else is set to 1
    image = numpy.expand_dims(image,axis=0)
    #image = image.transpose(2,0,1)
    #print(image.shape)
    return image.astype(numpy.float)

  def solve_environment(self):
    "The main interface between the agent and the environment"
    episode = 0
    epoch =0
    #Initialise the epoch arrays which are used for entropy calculations 
    epoch_logits = torch.empty(size=(0,self.env.action_space.n),device=self.device)
    epoch_weighted_log_probs = torch.empty(size=(0,),dtype=torch.float,device=self.device)
    
    while True:
      (episode_weighted_log_prob_trajectory,episode_logits,sum_of_episode_rewards,episode) = self.play_episode(episode=episode)
      #After each episode the rewards are added to the reward deque
      self.total_reward.append(sum_of_episode_rewards)
      #Append the weighted and logits arrays 
      epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs,episode_weighted_log_prob_trajectory),dim=0)
      epoch_logits = torch.cat((epoch_logits,episode_logits),dim=0)
      if episode >= self.batch_size :
        episode = 0
        epoch +=1
        loss,entropy = self.loss_calculator(epoch_logits=epoch_logits, weighted_log_probs=epoch_weighted_log_probs)
        self.optimiser.zero_grad()
        print("Clearing previous buffer",end='\n')
        loss.backward()
        print("Back propagating loss",end='\t')
        self.optimiser.step()
        self.scheduler.step(loss)
        print("Optimising ",end='\t')
        print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {numpy.mean(self.total_reward):.3f}",end="",flush=False)

        epoch_logits = torch.empty(size=(0,self.env.action_space.n),device=self.device)

        epoch_weighted_log_probs = torch.empty(size=(0,),dtype=torch.float,device=self.device)
        if(numpy.mean(self.total_reward)>150):
          print("\n Solved ")
          break
    self.env.close()
  def play_episode(self,episode:int):
    state = self.env.reset()
    previous_x = None
    episode_actions = torch.empty(size=(0,),dtype=torch.long,device=self.device)
    episode_logits = torch.empty(size=(0,self.env.action_space.n),device=self.device)
    average_rewards = numpy.empty(shape=(0,), dtype=numpy.float)
    episode_rewards = numpy.empty(shape=(0,), dtype=numpy.float)

    while True:
      #if not self.render:
      #  self.env.render()
      current_x = self.PreProcessing(state)
      x = current_x - previous_x if previous_x is not None else numpy.zeros_like(current_x)
      previous_x = current_x
      action_logits = self.agent(torch.tensor(x).float().unsqueeze(dim=0).to(self.device))
      episode_logits = torch.cat((action_logits,episode_logits),dim=0)
      action = Categorical(logits=action_logits).sample()
      episode_actions = torch.cat((episode_actions,action),dim=0)

      state,reward,done,_ = self.env.step(action = action.cpu().item())
      episode_rewards = numpy.concatenate((episode_rewards,numpy.array([reward])),axis=0)
      average_rewards = numpy.concatenate((average_rewards,numpy.expand_dims(numpy.mean(episode_rewards),axis=0)),axis=0)

      if done:
        episode+=1
        discounted_rewards = PG_RL.get_discounted_rewards(rewards=episode_rewards,gamma=self.gamma)
        discounted_rewards -= average_rewards
        discounted_rewards /= numpy.std(discounted_rewards)
        sum_of_rewards = numpy.sum(episode_rewards)
        mask = one_hot(episode_actions,num_classes=self.env.action_space.n)
        episode_log_probs = torch.sum(mask.float()*log_softmax(episode_logits,dim=1),dim=1)
        episode_weighted_log_probs = episode_log_probs * torch.tensor(discounted_rewards).float().to(self.device)
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)
        #show_video()
        return sum_weighted_log_probs, episode_logits, sum_of_rewards, episode
  
  def loss_calculator(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    policy_loss = -1*torch.mean(weighted_log_probs)
    p = softmax(epoch_logits, dim=1)
    log_p =log_softmax(epoch_logits, dim=1)
    entropy = -1*torch.mean(torch.sum(p*log_p,dim=1),dim=0)
    entropy_bonus = -1*self.beta*entropy
    policy_loss = policy_loss+entropy_bonus 
    return policy_loss, entropy

  @staticmethod
  def get_discounted_rewards(rewards: numpy.array, gamma:float) ->numpy.array :
    discounted_r = numpy.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0,rewards.size)):
      if rewards[t] !=0 : 
        running_add = 0
      running_add = running_add*gamma + rewards[t]
      discounted_r[t] = running_add
    return discounted_r
    def main():
  policy_gradient = PG_RL()
  policy_gradient.solve_environment()


if __name__ == "__main__":
    main()
