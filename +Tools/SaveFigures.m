function SaveFigures(ind)
for i = 1:numel(ind)
    figure(ind(i))
    savefig(strcat('./figures/',num2str(ind(i)),'.fig'))
end
end