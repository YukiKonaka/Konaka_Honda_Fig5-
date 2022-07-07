function particles = state_15_self_org(particles,a,o,C_noise) 

[numberOfStates, numberOfParticles] = size(particles);

dt=1;

%%particle 
ml=particles(1,:);
mr=particles(2,:);
pl=particles(3,:);
pr=particles(4,:);
C =particles(5,:);
am=particles(6,:);
vw=particles(7,:);

vw=particles(7,:);
Po=particles(8,:);

pw=1./vw;
fl=(1+exp(-1*ml)).^(-1); 
fr=(1+exp(-1*mr)).^(-1); 


dmldt=((1./pl)+vw).*(o-fl);
dmrdt=((1./pr)+vw).*(o-fr);


ml  =ml+am.*dmldt*dt*(a==1);
mr  =mr+am.*dmrdt*dt*(a==0);
Sl=(1+exp(-1*ml)).^(-1); 
Sr=(1+exp(-1*mr)).^(-1); 

pl=(((1./vw.*pl)./(pl+1./vw))+(Sl.*(1-Sl))).*(a==1)+((1./vw.*pl)./(pl+1./vw)).*(a==0);
pr=(((1./vw.*pr)./(pr+1./vw))+(Sr.*(1-Sr))).*(a==0)+((1./vw.*pr)./(pr+1./vw)).*(a==1);

C = C + C_noise*randn(1,numberOfParticles);

am = am;
vw = vw;

particles=[ml;mr;pl;pr;C;am;vw;Po];

end
