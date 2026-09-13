function step3b_R2_064()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'step3b_R2_064.log']);
  try
    m=mphload([base 'ARC_TA_loss_Nc12_geARC.mph']); fprintf('LOADED\n');
    fprintf('OLD R2=%s\n', char(m.param.get('R2')));
    m.param.set('R2','0.64[m]');           % WP 径向厚度 = FEM dr_tf
    Rin=m.param.evaluate('Rin'); R2=m.param.evaluate('R2'); Np=m.param.evaluate('Np');
    fprintf('NEW R2=0.64, 线圈 R=[%.2f, %.2f]\n', Rin, Rin+R2);
    fprintf('GEOM.RUN...\n'); m.geom('geom1').run; fprintf('GEOM_OK 域=%d 边界=%d\n', m.geom('geom1').getNDomains, m.geom('geom1').getNBoundaries);
    % 重设 PMC/MI (按位置, 防索引漂移)
    e=1e-3;
    pmc=mphselectbox(m,'geom1',[-e 6+e; -e e],'boundary');
    mi=unique([mphselectbox(m,'geom1',[6-e 6+e; -e 3+e],'boundary') mphselectbox(m,'geom1',[-e 6+e; 3-e 3+e],'boundary')]);
    m.physics('mf').feature('pmc1').selection.set(pmc); m.physics('mf').feature('mi2').selection.set(mi);
    fprintf('PMC=%s MI=%s\n', mat2str(pmc), mat2str(mi));
    % 验证选择集
    nb=numel(mphgetselection(m.component('comp1').selection('uni2')).entities);
    fprintf('uni2 带材线=%d, per-pancake=%.1f (期望94)\n', nb, nb/(Np/2));
    % 重做 ge8 (新 R2=0.64 ARC 互感)
    M=readmatrix([base 'ARC_M18.csv']); neq=size(M,1);
    eq=cell(neq,1);
    for i=1:neq
      t=''; for j=1:neq; t=[t sprintf('%.6e[H]*Ia%dt',M(i,j),j)]; if j<neq; t=[t ' + ']; end; end
      eq{i}=sprintf('current(t) - (Nt/Npw*Np)^2/(Rcoil*Np)*(%s) - Ia%d', t, i);
    end
    m.component('comp1').physics('ge').feature('ge8').set('equation', eq);
    beq=m.component('comp1').physics('ge').feature('ge8').getStringMatrix('equation'); b=char(beq(1));
    tok=regexp(b,'([\d.]+e-0\d)\[H\]\*Ia1t','tokens','once');
    fprintf('ge8 eq1 self=%s[H] (应~1.399e-05, R2=0.64 ARC)\n', tok{1});
    fprintf('MESH.RUN...\n'); m.mesh('mesh1').run; fprintf('MESH_OK 单元=%d\n', m.mesh('mesh1').getNumElem);
    mphsave(m,[base 'ARC_TA_loss_Nc12_R2.mph']);
    fprintf('SAVED ARC_TA_loss_Nc12_R2.mph\nR2_OK\n');
  catch ME
    fprintf(2,'R2_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
