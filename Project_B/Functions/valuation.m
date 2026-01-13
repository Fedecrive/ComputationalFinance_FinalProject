function val=valuation(n, F, i, j, k, disc, u, d, payoff, nsteps)
ricorsione=1
if n==1
    if j>=nsteps
        val=max(F(i,j)-k,0);
    else
        val=max(max(F(i,j)-k,0),disc(i)*(payoff(i,j+1)*u+payoff(i+1,j+1)*d));
    end
else
val=max(max(F(i,j)-k,0)+disc(i)*(valuation(n-1,F,i,j+1,k,disc,u,d,payoff,nsteps)*u+valuation(n-1,F,i+1,j+1,k,disc,u,d,payoff,nsteps)*d),...
    disc(i)*(valuation(n,F,i,j+1,k,disc,u,d,payoff,nsteps)*u+valuation(n,F,i+1,j+1,k,disc,u,d,payoff,nsteps)*d));
end
end