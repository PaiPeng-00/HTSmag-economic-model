function step4b_tighten_air()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'step4b_tighten_air.log']);
  try
    m=mphload([base 'ARC_TA_loss_Nc12_R2.mph']); fprintf('LOADED (36条ge8)\n');
    % 紧凑空气盒: R=[3,6] z=[0,1.2] (线圈 R=4.14-4.78 在内)
    m.geom('geom1').feature('r2').set('pos', [3 0]);
    m.geom('geom1').feature('r2').set('size', [3 1.2]);
    m.geom('geom1').run; fprintf('GEOM_OK 紧凑空气 R=[3,6] z=[0,1.2]\n');
    % 重设 BC: PMC=z=0; MI=R=3内 + R=6外 + z=1.2顶 (无轴)
    e=1e-3;
    pmc=mphselectbox(m,'geom1',[3-e 6+e; -e e],'boundary');
    mi=unique([mphselectbox(m,'geom1',[3-e 3+e; -e 1.2+e],'boundary') ...
               mphselectbox(m,'geom1',[6-e 6+e; -e 1.2+e],'boundary') ...
               mphselectbox(m,'geom1',[3-e 6+e; 1.2-e 1.2+e],'boundary')]);
    m.physics('mf').feature('pmc1').selection.set(pmc); m.physics('mf').feature('mi2').selection.set(mi);
    fprintf('PMC(z=0)=%s, MI(内/外/顶)=%s\n', mat2str(pmc), mat2str(mi));
    % 重网格, 报单元数对比
    m.mesh('mesh1').run; ne=m.mesh('mesh1').getNumElem;
    fprintf('MESH_OK 单元=%d (之前 R=0-6 是 47532; 期望大幅下降)\n', ne);
    mphsave(m,[base 'ARC_TA_loss_Nc12_R2.mph']);
    fprintf('SAVED. TIGHTEN_OK\n');
  catch ME
    fprintf(2,'TIGHTEN_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
