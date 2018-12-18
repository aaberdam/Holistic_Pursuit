% In this script we compare between two pursuit algorithms the Projection 
% algorithm [1] and the Holistic Pursuit algorithm [2]. 
% We consider the multi-layer sparse coding setup where the signal x can be
% sparse decomposed to a dictioanry D1 and a sparse representation gamma1;
% however, we assume that gamma1 has a sparse decomposition as well to D2
% and gamma2.
% 
% 
% [1] - Sulam, 2017, Multi-Layer Convolutional Sparse Modeling: Pursuit and Dictionary Learning
% [2] - Aberdam, 2018, Multi-Layer Sparse Coding: The Holistic Way
%% 0. Initialization
save_ind = 1; % Save variables
rng_idx = 1e7;
rng(rng_idx);

nitr = 1e2; % #iteration (should be >= 1e4)

n = 50; % Signal dimension
m1 = 100;% First layer dimension
m2 = 50;% Second layer dimension

l1 = (1:10); % Co-cardinality (= #zeros) of the first layer
s1 = m1 - l1; % Cardinality (= #nonzeros) of the first layer

% Dictionary_type:
% 0- Gaussian dictionary normalized columns,
% 1- Gaussian dictionary normalized columns of D1, normalized rows of D2
% 2- Diff matrix
Dictionary_type = 1;

% saveTrackHolistic:
% 0- The cardinality of the second layer is fixed 
% 1- The cardinality of the second layer relative to the co-cardinality of
% the first layer, such that s2-l1 is always equals to 1

% Both experiments are interesting! so run them both :)
saveTrackHolistic = 1;

if saveTrackHolistic == 0
    s2 = l1(end)+1;%l1+1; % l1+1;
else
    s2 = l1+1;
end

sigma_noise = 0.1;  % STD of the noise
snr_db = 25;%(0:5:20); % SNR
sig_norm = 10.^(snr_db/10) * sqrt(n) * sigma_noise; % The signal amplitude

%%% 0.1. Init variables
nVar = max([numel(l1), numel(snr_db), numel(s2)]); % Number of parameters
se_proj = zeros(nitr, nVar); % Squared Error using the projection approach 
supp_proj = zeros(nitr, nVar); % The ratio of support recovered succesfully using the projection algo
gamma1_squared_norm = zeros(nitr, nVar); % gamma1 energy
gamma2_squared_norm = zeros(nitr, nVar); % gamma2 energy

if saveTrackHolistic == 0
    se_holistic = zeros(nitr, nVar); % Squared Error using the holistic approach 
    supp_holistic = zeros(nitr, nVar); % Support recovery using the holistic approach 
else
    se_holistic = zeros(nitr, numel(l1), numel(s2)); % Squared Error using the holistic approach 
    supp_holistic = zeros(nitr, numel(l1), numel(s2)); % Support recovery using the holistic approach 
end

%%% 0.2. ML-SC model struct
MLSC_general = Tools.MLSC_init(n,m1,m2); % Init of Multi-Layer Sparse Coding struct
MLSC_general.dict_type = Dictionary_type; % Dictionary type as mentioned above

MLSC_general = CreateMlsrSignalD2Kernel(MLSC_general); % Create the dictionaries

%% 1. Main Loop
for iVar = 1:nVar
    %%% 1.1 Init for iVar
    
    % set the specific properties for current iteration
    MLSC_i = MLSC_general; % Init MLSC struct for this iteration
    MLSC_i.l1 = l1( min(iVar, numel(l1)) ); % set l1 (= #zeros in gamma1)
    s1_i = MLSC_i.m1 - MLSC_i.l1; % set s1 (= #nonzeros in gamma1)
    MLSC_i.s2 = s2( min( iVar, numel(s2)) ); % set s2 (= #nonzeros in gamma2)
    MLSC_i.sig_norm = sig_norm( min( iVar, numel(snr_db)) ); % set the signal norm
    
    % ADMM struct for the Basis Pursuit solver (=LASSO)
    % Details of the ADMM's fileds can be found in the
    % 'admm_lasso_constraint.m'
    ADMM_str = Tools.ADMM_init();
    ADMM_str.precision = 1; 
    ADMM_str.rho = 0.1;
    ADMM_str.lambda = 0.05;
    ADMM_str.admm_mode = 2;
    ADMM_str.thresh = MLSC_i.n * sigma_noise^2;
%     ADMM_str.admm_mode = 0;
%     ADMM_str.thresh = MLSC_i.s2;
    
    %%% 1.2 Inner Loop
    for itr = 1:nitr
        % Create the ML-SC siganl
        [MLSC_i, x, gamma1, gamma2, Lambda_1_c, Lambda_1, Lambda_2] = CreateMlsrSignalD2Kernel(MLSC_i);
        noise = sigma_noise * randn(n,1); % Create a noise vector
        y = x + noise; % y is the noisy vector
        
        % Layers norm
        gamma1_squared_norm(itr, iVar) = sum(gamma1.^2);
        gamma2_squared_norm(itr, iVar) = sum(gamma2.^2);
        
        % Projection Approach
        % Find an estimator for gamma2 using the projection approach
        [alpha2_proj, gamma2_proj, ADMM_str] = admm_lasso_constraint(y, MLSC_i.D_eff, eye(m2), ADMM_str);
        [se_proj(itr, iVar), supp_proj(itr, iVar)] = Tools.ErrorEvaluation(gamma2_proj, gamma2); % Accuracy of the estimation
        
        % Holistic
        % Find an estimator for gamma2 using the holistic approach
        [gamma2_holistic, gamma2_holistic_nz, Lambda_1_c_holistic] = HolisticPursuit(y, MLSC_i, ADMM_str, saveTrackHolistic);
        % Accuracy of the estimation
        if saveTrackHolistic == 0
            [se_holistic(itr, iVar), supp_holistic(itr, iVar)] = Tools.ErrorEvaluation(gamma2_holistic, gamma2);
        else
            [se_holistic(itr, (1: (MLSC_i.l1+1)), iVar), supp_holistic(itr, (1: (MLSC_i.l1+1)), iVar)] = Tools.ErrorEvaluation(gamma2_holistic, gamma2);
        end

    end
% fprintf('l1 = %d:  Projection %d, Holistic %d \n', l1_i, mean(se_proj(:,il1)),mean(se_holistic(:,il1)))
end

%% 2. Present Results
line_width = 6;
labels_fontsize = 22;
title_fontsize = 30;
legend_fontsize = 14;
axis_fontsize = 20;


if ~save_ind
    fig = figure('units','normalized','outerposition',[0.0 0 1 1]);
ax1 = axes('Position',[0 0 1 1],'Visible','off');
ax2 = axes('Position',[.3 .1 .6 .8]);
else 
    fig = figure('units','normalized','outerposition',[0.5 0.2 0.5 0.8]);
end
% title('Holistic Pursuit Vs Projection','FontSize',title_fontsize)
if saveTrackHolistic == 0
    if numel(l1) > 1
        xAx = l1;
    elseif numel(snr_db) > 1
        xAx = snr_db;
    elseif numel(s2) > 1
        xAx = s2;
    end
    
    hold on
    if save_ind
        plot(xAx, mean(se_proj, 1) , 'DisplayName', 'Projection', 'LineWidth', line_width)
        plot(xAx, mean(se_holistic, 1) , 'DisplayName', 'Holistic', 'LineWidth', line_width)
    else
        plot(ax2,xAx, mean(se_proj, 1) , 'DisplayName', 'Projection', 'LineWidth', line_width)
        plot(ax2,xAx, mean(se_holistic, 1) , 'DisplayName', 'Holistic', 'LineWidth', line_width)
    end
    hold off
    grid on
    tmp = xlabel('$\ell_1$','Interpreter','latex');
%     ylabel('$\frac{\left\Vert \gamma_2 - \hat{\gamma}_2 \right\Vert_2^2}{\left\Vert \gamma_2 \right\Vert_2^2}$','Interpreter','latex')
    ylabel('$\left\Vert \gamma_2 - \hat{\gamma}_2 \right\Vert_2^2 / \left\Vert \gamma_2 \right\Vert_2^2$','Interpreter','latex')
    legend('show','Location','southwest')
    
    h1 = tmp.Parent;
    h1.Legend.FontSize = legend_fontsize;
    h1.YScale = 'log';
    xlim([l1(1),l1(end)])
else
    if save_ind
        imagesc(s2, l1, squeeze(mean(se_holistic,1))); colormap(linspecer);
    else
        imagesc(ax2,s2, l1, squeeze(mean(se_holistic,1))); colormap(linspecer);
    end
    c = colorbar;
    c.FontSize = axis_fontsize
    tmp = xlabel('$s_2$','Interpreter','latex');
    ylabel('$\ell_1$','Interpreter','latex');
    title('$\frac{\left\Vert \gamma_2 - \hat{\gamma}_2 \right\Vert_2^2}{\left\Vert \gamma_2 \right\Vert_2^2}$','Interpreter','latex','FontSize',title_fontsize)
    h1 = tmp.Parent;
    axis equal
    xlim([s2(1),s2(end)])
    hold on
    contour(s2,[0,l1],squeeze(mean(se_holistic,1)),'k','LineWidth',1,'LineStyle','--');
    hold off
end

h1.XAxis.FontSize = axis_fontsize;
h1.YAxis.FontSize = axis_fontsize;
h1.XLabel.FontSize = labels_fontsize;
h1.YLabel.FontSize = labels_fontsize;


if ~save_ind
dict_type_snr = {'Gaussian dictionary \n with normalized columns',...
    'Gaussian dictionary \n normalized columns of D1, \n normalized rows of D2',...
    'Diff matrix',...
    'D1 - Gaussian dictionary, D2- Sparse matrix'};

descr = {'With the values:';
    sprintf('n = %d',n);
    sprintf('m1 = %d',m1);
    sprintf('m2 = %d',m2);
    [sprintf('l1:'),sprintf('%d, ',l1)];
    [sprintf('s1:'),sprintf('%d, ',s1)];
    [sprintf('s2:'),sprintf('%d, ',s2)];
    [sprintf('SNR: '), sprintf('%d[dB], ',snr_db)];
    sprintf('#Iterations = %d ',nitr);
    sprintf(dict_type_snr{Dictionary_type+1});
    sprintf('rng %d\n',rng_idx);
    };
axes(ax1) % sets ax1 to current axes
text(.025,0.6,descr,'FontSize',16)
end

if save_ind
%     saveas(gcf,strcat('./figures/HolisticVsProjection/','HolisticPursuit',num2str(snr_db),'snr'),'epsc')
%     print(strcat('./figures/HolisticVsProjection/','HolisticPursuit',num2str(snr_db),'snr'),'-depsc','-r0')
%     dt = datestr(now,'mm_dd_HH_MM');
%     savefig(strcat('./figures/HolisticVsProjection/',dt,'.fig'))
%     save(strcat('./figures/HolisticVsProjection/',dt))
end
