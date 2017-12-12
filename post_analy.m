% This script is for post analysis
% generates figures 4~6

res_files = dir('result_figures/*png');
for rf=1:length(res_files)
    res = split(res_files(rf).name(1:end-4),'_');
    if length(res)==10
        size(rf) = str2num(res{4});
        order(rf) = str2num(res{6});
        delta(rf) = str2num(res{8});
        
        I_tru = imread(['test_',num2str(size(rf)),'.png']);
        I_res = imread(fullfile(res_files(rf).folder,res_files(rf).name))>127;
        err(rf) = nnz(I_tru-I_res)/numel(I_tru);%str2num(res{10});
    end

end
    
%% error vs order
figure
for su=unique(size)
    subplot(1,4,su/50), hold on, grid on, box on
    ss = size==su;

    for du=unique(delta)
        dd= delta==du;
        plot_pairs = sortrows([order(dd.*ss~=0)',err(dd.*ss~=0)']);
        plot(plot_pairs(:,1),plot_pairs(:,2),'o-','LineWidth',2,'MarkerSize',10);
    end
    title(['image size = ',num2str(su)])
    xlabel('order (context length)')
    ylabel('error rate')
    axis([8,20,0,0.08])
end
legend({'delta = .01','delta = .02','delta = .05','delta = .10'})

%% error vs delta 
figure
for su=unique(size)
    subplot(1,4,su/50), hold on, grid on, box on
    ss = size==su;

    for ou=unique(order)
        oo= order==ou;
        plot_pairs = sortrows([delta(ss.*oo~=0)',err(ss.*oo~=0)']);
        plot(plot_pairs(:,1),plot_pairs(:,2),'o-','LineWidth',2,'MarkerSize',10);
    end
    title(['image size = ',num2str(su)])
    xlabel('delta (BSC parameter)')
    ylabel('error rate')
    ylim([0,0.08])
end
legend({'order = 8','order = 9','order = 10','order = 11','order = 12','order = 14','order = 16','order = 18','order = 20'},'Location','NorthWest')

%% error vs size
figure
for du=unique(delta)
    subplot(1,4,find(du==unique(delta))), hold on, grid on, box on
    dd = delta==du;

    for ou=unique(order)
        oo= order==ou;
        plot_pairs = sortrows([size(dd.*oo~=0)',err(dd.*oo~=0)']);
        plot(plot_pairs(:,1),plot_pairs(:,2),'o-','LineWidth',2,'MarkerSize',10);
    end
    title(['delta = ',num2str(du)])
    xlabel('image size')
    ylabel('error rate')
    %ylim([0,0.08])
end
legend({'order = 8','order = 9','order = 10','order = 11','order = 12','order = 14','order = 16','order = 18','order = 20'},'Location','NorthWest')



[tpr,fpr,thresholds] = roc(I_tru(:),I_res(:));
plotroc(I_tru(:),I_res(:));