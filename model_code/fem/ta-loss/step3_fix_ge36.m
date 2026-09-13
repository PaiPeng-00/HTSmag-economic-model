function step3_fix_ge36()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'step3_fix_ge36.log']);
  try
    % 从完整 ge8 的 air 模型重来 (它有 36 条方程)
    m=mphload([base 'ARC_TA_loss_Nc12_air.mph']); fprintf('LOADED air (完整36条ge8)\n');
    m.param.set('R2','0.64[m]');
    m.geom('geom1').run; fprintf('GEOM_OK R2=0.64, 线圈R=[%.2f,%.2f]\n', m.param.evaluate('Rin'), m.param.evaluate('Rin')+m.param.evaluate('R2'));
    e=1e-3;
    pmc=mphselectbox(m,'geom1',[-e 6+e; -e e],'boundary');
    mi=unique([mphselectbox(m,'geom1',[6-e 6+e; -e 3+e],'boundary') mphselectbox(m,'geom1',[-e 6+e; 3-e 3+e],'boundary')]);
    m.physics('mf').feature('pmc1').selection.set(pmc); m.physics('mf').feature('mi2').selection.set(mi);
    % --- 原地改 ge8: 只改含互感的 Ia 方程, 保留 Ir 方程 ---
    M=readmatrix([base 'ARC_M18.csv']);   % R2=0.64 ARC 18x18
    ge=m.component('comp1').physics('ge').feature('ge8');
    arr=ge.getStringMatrix('equation'); neq=numel(arr);
    fprintf('原 ge8 方程数=%d (应36)\n', neq);
    newarr=cell(neq,1); nIa=0; nIr=0;
    for k=1:neq
      s=char(arr(k));
      if ~isempty(strfind(s,'(Rcoil*Np)*('))     % Ia 方程
        i=str2double(regexp(s,'- Ia(\d+)\s*$','tokens','once'));
        t=''; for j=1:18; t=[t sprintf('%.6e[H]*Ia%dt',M(i,j),j)]; if j<18; t=[t ' + ']; end; end
        newarr{k}=sprintf('current(t) - (Nt/Npw*Np)^2/(Rcoil*Np)*(%s) - Ia%d', t, i); nIa=nIa+1;
      else                                        % Ir 方程, 不动
        newarr{k}=s; nIr=nIr+1;
      end
    end
    ge.set('equation', newarr);
    fprintf('改了 %d 个 Ia 方程, 保留 %d 个 Ir 方程, 共 %d\n', nIa, nIr, nIa+nIr);
    % 验证
    bk=ge.getStringMatrix('equation');
    fprintf('eq1: %s\n', char(bk(1)));
    fprintf('eq19(应Ir1): %s\n', char(bk(19)));
    tok=regexp(char(bk(1)),'([\d.]+e-0\d)\[H\]\*Ia1t','tokens','once');
    fprintf('eq1 self=%s (应1.399e-05 ARC)\n', tok{1});
    m.mesh('mesh1').run;
    mphsave(m,[base 'ARC_TA_loss_Nc12_R2.mph']);
    fprintf('SAVED (覆盖 R2.mph, 现含完整36条). FIX36_OK\n');
  catch ME
    fprintf(2,'FIX36_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
