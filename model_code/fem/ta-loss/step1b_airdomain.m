function step1b_airdomain()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'step1b_airdomain.log']);
  try
    m=mphload([base 'ARC_TA_loss_Nc12_step1.mph']); fprintf('LOADED\n');
    % --- 放大空气域 r2: R=[0,6] z=[0,3] (含轴) ---
    m.geom('geom1').feature('r2').set('pos', [0 0]);
    m.geom('geom1').feature('r2').set('size', [6 3]);
    fprintf('GEOM.RUN (air R=[0,6] z=[0,3])...\n'); m.geom('geom1').run;
    fprintf('GEOM_OK 域=%d 边界=%d\n', m.geom('geom1').getNDomains, m.geom('geom1').getNBoundaries);
    % --- 按位置找边界重设 PMC/MI ---
    e=1e-3;
    pmc = mphselectbox(m,'geom1',[-e 6+e; -e e],'boundary');        % z=0 对称面
    mir = mphselectbox(m,'geom1',[6-e 6+e; -e 3+e],'boundary');     % R=6 外
    mit = mphselectbox(m,'geom1',[-e 6+e; 3-e 3+e],'boundary');     % z=3 顶
    mi  = unique([mir mit]);
    fprintf('PMC(z=0) 边界: %s (%d个)\n', mat2str(pmc), numel(pmc));
    fprintf('MI(R=6/z=3) 边界: %s (%d个)\n', mat2str(mi), numel(mi));
    if isempty(pmc)||isempty(mi); fprintf(2,'!! 边界定位失败, 不重设\n'); else
      m.physics('mf').feature('pmc1').selection.set(pmc);
      m.physics('mf').feature('mi2').selection.set(mi);
      fprintf('PMC/MI 已按位置重设\n');
    end
    % --- mesh ---
    fprintf('MESH.RUN...\n'); tm=tic; m.mesh('mesh1').run; fprintf('MESH_OK %.0fs, 单元=%d\n', toc(tm), m.mesh('mesh1').getNumElem);
    mphsave(m,[base 'ARC_TA_loss_Nc12_air.mph']);
    fprintf('SAVED ARC_TA_loss_Nc12_air.mph\nAIR_OK\n');
  catch ME
    fprintf(2,'AIR_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
