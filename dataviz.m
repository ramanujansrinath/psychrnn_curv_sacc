clc; clear
load('/Volumes/colada/Ram/psychrnn_test/try3/data/test_model1.mat')

%%
params = cell2mat(cellfun(@(x) [x.s double(x.a) double(x.b) double(x.o)]',trial_params,'UniformOutput',false))';

arc_conds = unique(params(:,2:3),'rows');
clf;
for ii=1:size(arc_conds,1)
    subplot(121); hold on;
    plot(...
        params(params(:,2)==arc_conds(ii,1) & params(:,3)==arc_conds(ii,2),1),...
        params(params(:,2)==arc_conds(ii,1) & params(:,3)==arc_conds(ii,2),4),...
        '.');
    subplot(122); hold on;
    correct_sacc = cellfun(@(x) x.o,trial_params);
    actual_sacc = outputs(:,end);
    plot(correct_sacc(params(:,2)==arc_conds(ii,1) & params(:,3)==arc_conds(ii,2)),...
        actual_sacc(params(:,2)==arc_conds(ii,1) & params(:,3)==arc_conds(ii,2)),...
        '.')
end
fixPlot(subplot(121),[-0.1 1.1],[-90 90],'curvature','saccade target',0:0.25:1,-90:45:90,'inference to expected behaviour')
fixPlot(subplot(122),[-90 90],[-90 90],'saccade target','actual saccade',-90:45:90,-90:45:90,'behaviour')

%%
for ii=1:50
    clf; subplot(121); hold on;
    plot(test_inputs(ii,:,1))
    plot(test_inputs(ii,:,2)/180)
    plot(test_inputs(ii,:,3)/180)
    plot(test_inputs(ii,:,4))
    plot(target_outputs(ii,:,1)/180,'linewidth',2)
%     plot(target_outputs(ii,:,2),'linewidth',2)
    line([trial_params{ii}.fix_onset trial_params{ii}.fix_onset]/10,[0 1])
    line([trial_params{ii}.s_onset trial_params{ii}.s_onset]/10,[0 1])
    line([trial_params{ii}.ab_onset trial_params{ii}.ab_onset]/10,[0 1])
    line([trial_params{ii}.fix_offset trial_params{ii}.fix_offset]/10,[0 1])


    subplot(122); hold on;
    plot(target_outputs(ii,:,1)/180,'linewidth',2)
%     plot(target_outputs(ii,:,2),'linewidth',2)
    plot(outputs(ii,:,1)/180,'linewidth',2)
%     plot(outputs(ii,:,2),'linewidth',2)
    line([trial_params{ii}.fix_onset trial_params{ii}.fix_onset]/10,[0 1])
    line([trial_params{ii}.s_onset trial_params{ii}.s_onset]/10,[0 1])
    line([trial_params{ii}.ab_onset trial_params{ii}.ab_onset]/10,[0 1])
    line([trial_params{ii}.fix_offset trial_params{ii}.fix_offset]/10,[0 1])
    pause;
end

%%
nComp = 4; dimIDs = 2:4;
traj = reshape(pca(reshape(state_vars,[1000*180 50])','NumComponents',nComp),[1000 180 nComp]);
curv = cellfun(@(x) x.s,trial_params);
targ = cellfun(@(x) x.o,trial_params);
binE = linspace(0,1,11);
binC = (binE+circshift(binE,-1))/2; binC = binC(1:end-1);
clf;
cols = lines(size(arc_conds,1));
for ii=1:size(arc_conds,1)
    trials = cell2mat(trial_params(params(:,2)==arc_conds(ii,1) & params(:,3)==arc_conds(ii,2)));

    stat_ii = state_vars(params(:,2)==arc_conds(ii,1) & params(:,3)==arc_conds(ii,2),:,:);
    traj_ii = traj(params(:,2)==arc_conds(ii,1) & params(:,3)==arc_conds(ii,2),:,:);
    curv_ii = [trials.s];
    targ_ii = [trials.o];
    sel_ii = outputs(params(:,2)==arc_conds(ii,1) & params(:,3)==arc_conds(ii,2),end);
    
    traj_ii = traj_ii(:,:,dimIDs);

    % plot all trials
    subplot(321); hold on;
    for jj=1:size(traj_ii,1)
        plot3(...
            traj_ii(jj,:,1),traj_ii(jj,:,2),traj_ii(jj,:,3),...
            'color',[curv_ii(jj) 0 0],'linewidth',0.5);
        tp = round(trials(jj).fix_onset/10);
        plot3(...
            traj_ii(jj,tp,1),traj_ii(jj,tp,2),traj_ii(jj,tp,3),'k.');
        tp = round(trials(jj).s_onset/10);
        plot3(...
            traj_ii(jj,tp,1),traj_ii(jj,tp,2),traj_ii(jj,tp,3),'k.');
        tp = round(trials(jj).ab_onset/10);
        plot3(...
            traj_ii(jj,tp,1),traj_ii(jj,tp,2),traj_ii(jj,tp,3),'k.');
        tp = round(trials(jj).fix_offset/10);
        plot3(...
            traj_ii(jj,tp,1),traj_ii(jj,tp,2),traj_ii(jj,tp,3),'k.');
        
    end

    % plot after binning by curvatures
    subplot(322); hold on;
    [~,~,idx] = histcounts(curv_ii,binE);
    for jj=1:length(binC)
        traj_jj = squeeze(mean(traj_ii(idx==jj,:,:)));
        plot3(...
            traj_jj(:,1),traj_jj(:,2),traj_jj(:,3),...
            'color',binC(jj)*cols(ii,:));
    end

    % resp = squeeze(traj_ii(:,end,:));
    resp = squeeze(stat_ii(:,end,:));

    % subplot(234); hold on;
    % pred = decodeFn(sel_ii,resp);
    % plot(sel_ii,pred,'.','color',cols(ii,:))

    subplot(323); hold on;
    plot(curv_ii,targ_ii,'.','color',cols(ii,:))

    subplot(324); hold on;
    plot(curv_ii,sel_ii,'.','color',cols(ii,:))

    subplot(349); hold on;
    pred = decodeFn(curv_ii',resp);
    plot(curv_ii,pred,'.','color',cols(ii,:))

    subplot(3,4,10); hold on;
    pred = decodeFn(targ_ii',resp);
    plot(targ_ii,pred,'.','color',cols(ii,:)) 

    subplot(3,4,11); hold on;
    traj_ii = traj(params(:,2)==arc_conds(ii,1) & params(:,3)==arc_conds(ii,2),:,:);
    resp = squeeze(traj_ii(:,end,dimIDs));
%     resp = squeeze(traj_ii(:,end,:));
    pred = decodeFn(targ_ii',resp);
    plot(targ_ii,pred,'.','color',cols(ii,:)) 

    subplot(3,4,12); hold on;
    pred_t = decodeFn(targ_ii',resp);
    plot(curv_ii,pred_t,'.','color',cols(ii,:)) 
end

fix3dPlot(subplot(321),[],[],[],'PC1','PC2','PC3')
fix3dPlot(subplot(322),[],[],[],'PC1','PC2','PC3')
fixPlot(subplot(323),[-0.1 1.1],[-90 90],'curvature','correct target',0:0.25:1,-90:45:90,'ideal')
fixPlot(subplot(324),[-0.1 1.1],[-90 90],'curvature','selected target',0:0.25:1,-90:45:90,'behaviour')
fixPlot(subplot(349),[-0.1 1.1],[-0.1 1.1],'curvature','decoded curvature',0:0.25:1,0:0.25:1,'curvature linear decoding')
fixPlot(subplot(3,4,10),[-90 90],[-90 90],'target','decoded target',-90:45:90,-90:45:90,'target linear decoding (full dim)')
fixPlot(subplot(3,4,11),[-90 90],[-90 90],'target','decoded target',-90:45:90,-90:45:90,'target linear decoding (3 dim)')
fixPlot(subplot(3,4,12),[-0.1 1.1],[-90 90],'curvature','decoded target',0:0.25:1,-90:45:90,'curvature and decoded target (3 dim)')
