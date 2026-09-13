function step4_smoke()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'step4_smoke.log']);
  try
    m=mphload([base 'ARC_TA_loss_Nc12_R2.mph']); fprintf('LOADED %s\n', datestr(now,31));
    m.param('par6').set('T','10[K]'); m.param('par6').set('Ip','583');
    m.param('par2').set('Npw','10'); m.param('par7').set('rho_turn','5000[uohm*cm^2]');
    fprintf('单点: Npw=10 T=10K Ip=583 R2=%s\n', char(m.param.get('R2')));
    m.study('std1').feature('param').active(false);
    m.study('std1').feature('param1').active(false);
    m.study('std1').feature('time').set('tlist','range(0,td/4,td)');
    m.geom('geom1').run; m.mesh('mesh1').run;
    fprintf('SOLVING... %s\n', datestr(now,31)); ts=tic; m.sol('sol1').runAll; fprintf('SOLVED %.0fs (%s)\n', toc(ts), datestr(now,31));
    % --- int4 磁化损耗 ---
    try; m.result.table.create('tmag','Table'); catch; end
    m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1'); m.result.numerical('int4').setResult;
    Dm=mphtable(m,'tmag').data; magpk=max(Dm(:,2));
    % --- gev9 径向损耗 total ---
    try; m.result.table.create('trad','Table'); catch; end
    m.result.numerical('gev9').set('table','trad'); m.result.numerical('gev9').set('data','dset1'); m.result.numerical('gev9').setResult;
    Dr=mphtable(m,'trad').data; radpk=max(Dr(:,2));
    % --- 场 sanity check ---
    Bmax=NaN; try; Bmax=mphmax(m,'mf.normB','volume','dataset','dset1'); catch; end
    fprintf('=== 第4步冒烟结果 (R2=0.64, ARC ge8, Npw10/T10K/Ip583) ===\n');
    fprintf('磁化损耗 int4 vs t: '); fprintf('%.3f ', Dm(:,2)); fprintf('W/coil, peak=%.4f\n', magpk);
    fprintf('径向损耗 gev9 vs t: '); fprintf('%.1f ', Dr(:,2)); fprintf('W/coil, peak=%.2f\n', radpk);
    fprintf('场 max|B|=%.2f T (sanity: 有限非零=空气域/BC OK)\n', Bmax);
    writematrix([Dm(:,1) Dm(:,2) Dr(:,2)], [res 'ARC_step4_smoke_Npw10_T10_Ip583.csv']);
    mphsave(m,[base 'ARC_TA_loss_Nc12_R2_solved.mph']);
    fprintf('SAVED solved. STEP4_OK %s\n', datestr(now,31));
  catch ME
    fprintf(2,'STEP4_ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
