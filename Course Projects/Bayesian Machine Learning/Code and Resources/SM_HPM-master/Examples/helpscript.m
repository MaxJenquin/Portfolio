hypcontainer=[];
for k=1:100
    try
        hypcontainer = [hypcontainer; KDV_test(end-1:end)];
    catch
        fprintf('matrix was singular, at k=%d\n',k);
    end
end